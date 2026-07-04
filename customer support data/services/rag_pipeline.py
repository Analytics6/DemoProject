"""End-to-end LangChain RAG pipeline: ingest → preprocess → chunk → embed → index → retrieve → generate."""

import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain_bootstrap import ensure_stdlib_logging

ensure_stdlib_logging()

from langchain.chains import RetrievalQA
from langchain_core.documents import Document

from augmentation.Fulltext import build_full_text_documents, build_ticket_documents
from augmentation.chunking import chunk_documents
from augmentation.topkresults import get_top_k_results
from augmentation.vectordb import LangChainVectorDB
from generation.llm import LLMModels
from generation.prompt import Prompting
from retrieval.Preprocessing import (
    preprocess_faq,
    preprocess_inventory,
    preprocess_promotions,
    preprocess_tickets,
)

INDEX_DIR = ROOT / "databases" / "chroma_db"
MANIFEST_PATH = INDEX_DIR / "manifest.json"


@dataclass
class PipelineStats:
    inventory_docs: int = 0
    promotion_docs: int = 0
    faq_docs: int = 0
    ticket_docs: int = 0
    total_source_docs: int = 0
    total_chunks: int = 0
    indexed_at: str = ""
    embedding_model: str = "fake"
    index_path: str = ""
    categories: Dict[str, int] = field(default_factory=dict)


class EndToEndRAGPipeline:
    """
    Full RAG pipeline:
      1. Ingest   — load inventory, promotions, FAQ, 500 tickets
      2. Preprocess — clean and normalize all sources
      3. Chunk    — RecursiveCharacterTextSplitter (LangChain)
      4. Embed    — OpenAI embeddings (or FakeEmbeddings fallback)
      5. Index    — ChromaDB vector store (persisted to disk)
      6. Retrieve — similarity search + strategy routing
      7. Generate — RetrievalQA chain with LangChain LLM
    """

    STRATEGIES = ["naive_rag", "hybrid_rag", "graph_rag", "agentic_rag"]

    def __init__(
        self,
        data_dir: Optional[str] = None,
        api_key: Optional[str] = None,
        chunk_size: int = 280,
        chunk_overlap: int = 50,
    ) -> None:
        self.data_dir = Path(data_dir or ROOT / "data")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.llm = LLMModels(api_key=self.api_key)
        self.prompter = Prompting()
        self.vector_db = LangChainVectorDB(
            api_key=self.api_key,
            persist_directory=str(INDEX_DIR),
        )
        self._all_chunks: List[Document] = []
        self._source_docs: List[Document] = []
        self.stats = PipelineStats()
        self._setup()

    def _setup(self) -> None:
        manifest = self._load_manifest()
        if manifest and self._try_load_index(manifest):
            return
        self.build_index()

    def ingest(self) -> Dict[str, List[Dict]]:
        tickets_path = self.data_dir / "tickets.json"
        if not tickets_path.exists():
            from scripts.generate_tickets import generate_tickets
            tickets_path.write_text(
                json.dumps(generate_tickets(500), indent=2), encoding="utf-8"
            )
        return {
            "inventory": preprocess_inventory(str(self.data_dir / "inventory.csv")),
            "promotions": preprocess_promotions(str(self.data_dir / "promotions.json")),
            "faq": preprocess_faq(str(self.data_dir / "faq.json")),
            "tickets": preprocess_tickets(str(tickets_path)),
        }

    def preprocess_and_chunk(self, raw: Dict[str, List[Dict]]) -> List[Document]:
        docs = build_full_text_documents(raw["inventory"], raw["promotions"], raw["faq"])
        ticket_docs = build_ticket_documents(raw["tickets"])
        docs.extend(ticket_docs)
        self._source_docs = docs
        self.stats.inventory_docs = len(raw["inventory"])
        self.stats.promotion_docs = len(raw["promotions"])
        self.stats.faq_docs = len(raw["faq"])
        self.stats.ticket_docs = len(raw["tickets"])
        self.stats.total_source_docs = len(docs)
        self.stats.categories = {}
        for t in raw["tickets"]:
            cat = t.get("category", "general")
            self.stats.categories[cat] = self.stats.categories.get(cat, 0) + 1
        chunks = chunk_documents(docs, chunk_size=self.chunk_size, overlap=self.chunk_overlap)
        self._all_chunks = chunks
        self.stats.total_chunks = len(chunks)
        return chunks

    def build_index(self) -> PipelineStats:
        raw = self.ingest()
        chunks = self.preprocess_and_chunk(raw)
        LangChainVectorDB.clear_persist_directory(str(INDEX_DIR))
        self.vector_db = LangChainVectorDB(
            api_key=self.api_key,
            persist_directory=str(INDEX_DIR),
        )
        self.vector_db.upsert_documents(chunks)
        self.vector_db.save(str(INDEX_DIR))
        self.stats.indexed_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        self.stats.embedding_model = self.vector_db.embedding_model_name
        self.stats.index_path = str(INDEX_DIR)
        self._save_manifest()
        return self.stats

    def rebuild_index(self) -> PipelineStats:
        return self.build_index()

    def _save_manifest(self) -> None:
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        MANIFEST_PATH.write_text(
            json.dumps(
                {
                    "indexed_at": self.stats.indexed_at,
                    "total_chunks": self.stats.total_chunks,
                    "total_source_docs": self.stats.total_source_docs,
                    "ticket_docs": self.stats.ticket_docs,
                    "embedding_model": self.stats.embedding_model,
                    "vector_store": "chromadb",
                    "chunk_size": self.chunk_size,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _load_manifest(self) -> Optional[Dict]:
        if not MANIFEST_PATH.exists():
            return None
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    def _try_load_index(self, manifest: Dict) -> bool:
        chroma_path = INDEX_DIR
        if not chroma_path.exists() or manifest.get("vector_store") not in (None, "chromadb"):
            return False
        try:
            self.vector_db = LangChainVectorDB(
                api_key=self.api_key,
                persist_directory=str(INDEX_DIR),
            )
            self.vector_db.load(str(INDEX_DIR))
            if self.vector_db.document_count() == 0:
                return False
            raw = self.ingest()
            docs = build_full_text_documents(raw["inventory"], raw["promotions"], raw["faq"])
            docs.extend(build_ticket_documents(raw["tickets"]))
            self._source_docs = docs
            self._all_chunks = chunk_documents(
                docs, chunk_size=self.chunk_size, overlap=self.chunk_overlap
            )
            self.stats.total_chunks = manifest.get("total_chunks", len(self._all_chunks))
            self.stats.total_source_docs = manifest.get("total_source_docs", len(docs))
            self.stats.ticket_docs = manifest.get("ticket_docs", 0)
            self.stats.indexed_at = manifest.get("indexed_at", "")
            self.stats.embedding_model = manifest.get("embedding_model", "fake")
            self.stats.index_path = str(INDEX_DIR)
            raw_tickets = raw["tickets"]
            self.stats.categories = {}
            for t in raw_tickets:
                cat = t.get("category", "general")
                self.stats.categories[cat] = self.stats.categories.get(cat, 0) + 1
            return True
        except Exception:
            return False

    def get_pipeline_status(self) -> Dict:
        return {
            "stages": [
                {"id": "ingest", "label": "Ingest", "status": "complete", "count": self.stats.total_source_docs},
                {"id": "preprocess", "label": "Preprocess", "status": "complete", "count": self.stats.total_source_docs},
                {"id": "chunk", "label": "Chunk", "status": "complete", "count": self.stats.total_chunks},
                {"id": "embed", "label": "Embed", "status": "complete", "model": self.stats.embedding_model},
                {"id": "index", "label": "Index (ChromaDB)", "status": "complete", "path": self.stats.index_path},
                {"id": "retrieve", "label": "Retrieve", "status": "ready"},
                {"id": "generate", "label": "Generate", "status": "ready"},
            ],
            "stats": {
                "inventory_docs": self.stats.inventory_docs,
                "promotion_docs": self.stats.promotion_docs,
                "faq_docs": self.stats.faq_docs,
                "ticket_docs": self.stats.ticket_docs,
                "total_source_docs": self.stats.total_source_docs,
                "total_chunks": self.stats.total_chunks,
                "indexed_at": self.stats.indexed_at,
                "embedding_model": self.stats.embedding_model,
                "vector_store": "chromadb",
                "ticket_categories": self.stats.categories,
            },
        }

    def retrieve(self, query: str, top_k: int = 4, doc_type: Optional[str] = None) -> List[Dict]:
        if doc_type:
            filtered = LangChainVectorDB(
                api_key=self.api_key,
                collection_name=f"filter_{doc_type}",
            )
            typed = [c for c in self._all_chunks if c.metadata.get("type") == doc_type]
            filtered.upsert_documents(typed or self._all_chunks)
            return get_top_k_results(filtered, query=query, top_k=top_k)
        return get_top_k_results(self.vector_db, query=query, top_k=top_k)

    def _run_chain(self, query: str, retriever, prompt, model: str) -> str:
        if retriever is None:
            return "No context found in the knowledge base."
        chain = RetrievalQA.from_chain_type(
            llm=self.llm.get_llm(model),
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=False,
        )
        return chain.invoke({"query": query})["result"]

    def naive_rag(self, query: str, top_k: int = 4, model: str = "openai") -> Dict:
        ctx = self.retrieve(query, top_k=top_k)
        answer = self._run_chain(
            query,
            self.vector_db.as_retriever(top_k=top_k),
            self.prompter.singleshot_prompting_template(),
            model,
        )
        return {"strategy": "naive_rag", "answer": answer, "context": ctx}

    def hybrid_rag(self, query: str, top_k: int = 4, model: str = "openai") -> Dict:
        keywords = query.lower().split()
        filtered = [
            c for c in self._all_chunks
            if any(kw in c.page_content.lower() for kw in keywords)
        ]
        db = LangChainVectorDB(api_key=self.api_key, collection_name="hybrid_temp")
        db.upsert_documents(filtered or self._all_chunks)
        ctx = get_top_k_results(db, query=query, top_k=top_k)
        answer = self._run_chain(
            query, db.as_retriever(top_k=top_k),
            self.prompter.multishot_prompting_template(), model,
        )
        return {"strategy": "hybrid_rag", "answer": answer, "context": ctx}

    def graph_rag(self, query: str, top_k: int = 4, model: str = "openai") -> Dict:
        intent = self._detect_intent(query)
        typed = [c for c in self._all_chunks if c.metadata.get("type") == intent]
        db = LangChainVectorDB(api_key=self.api_key, collection_name="graph_temp")
        db.upsert_documents(typed or self._all_chunks)
        ctx = get_top_k_results(db, query=query, top_k=top_k)
        answer = self._run_chain(
            query, db.as_retriever(top_k=top_k),
            self.prompter.singleshot_prompting_template(), model,
        )
        return {"strategy": "graph_rag", "intent": intent, "answer": answer, "context": ctx}

    def agentic_rag(self, query: str, top_k: int = 4, model: str = "openai") -> Dict:
        intent = self._detect_intent(query)
        if intent == "promotions":
            result = self.hybrid_rag(query, top_k=top_k, model=model)
        elif intent in ("inventory", "ticket"):
            result = self.graph_rag(query, top_k=top_k, model=model)
        else:
            result = self.naive_rag(query, top_k=top_k, model=model)
        result["strategy"] = "agentic_rag"
        result["intent"] = intent
        return result

    def query(self, text: str, strategy: str = "agentic_rag", model: str = "openai", top_k: int = 4) -> Dict:
        handler = getattr(self, strategy, self.agentic_rag)
        return handler(text, top_k=top_k, model=model)

    @staticmethod
    def _detect_intent(query: str) -> str:
        q = query.lower()
        if any(t in q for t in ["ticket", "refund", "order", "complaint", "case", "support"]):
            return "ticket"
        if any(t in q for t in ["stock", "inventory", "available", "price"]):
            return "inventory"
        if any(t in q for t in ["deal", "discount", "offer", "promotion", "sale"]):
            return "promotions"
        return "general"


# Backward-compatible alias
RagModel = EndToEndRAGPipeline
