from pathlib import Path
from typing import Dict, List

from langchain_bootstrap import ensure_stdlib_logging

ensure_stdlib_logging()

from langchain.chains import RetrievalQA
from langchain_core.documents import Document

from augmentation.Fulltext import build_full_text_documents
from augmentation.chunking import chunk_documents
from augmentation.topkresults import get_top_k_results
from augmentation.vectordb import LangChainVectorDB
from generation.llm import LLMModels
from generation.prompt import Prompting
from retrieval.Preprocessing import preprocess_faq, preprocess_inventory, preprocess_promotions


class RagModel:
    def __init__(self, data_dir: str = "data") -> None:
        self.data_dir = Path(data_dir)
        self.vector_db = LangChainVectorDB()
        self.prompter = Prompting()
        self.llm = LLMModels()
        self._all_chunks: List[Document] = []
        self._setup()

    def _setup(self) -> None:
        inventory = preprocess_inventory(str(self.data_dir / "inventory.csv"))
        promotions = preprocess_promotions(str(self.data_dir / "promotions.json"))
        faq = preprocess_faq(str(self.data_dir / "faq.json"))

        docs = build_full_text_documents(inventory, promotions, faq)
        self._all_chunks = chunk_documents(docs, chunk_size=220, overlap=40)
        self.vector_db.upsert_documents(self._all_chunks)

    def naive_rag(self, query: str, top_k: int = 3, model: str = "openai") -> Dict:
        top_results = get_top_k_results(self.vector_db, query=query, top_k=top_k)
        retriever = self.vector_db.as_retriever(top_k=top_k)
        if retriever is None:
            return {"strategy": "naive_rag", "answer": "No context found.", "context": []}

        chain = RetrievalQA.from_chain_type(
            llm=self.llm.get_model(model),
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": self.prompter.singleshot_prompting_template()},
            return_source_documents=False,
        )
        answer = chain.invoke({"query": query})["result"]
        return {"strategy": "naive_rag", "answer": answer, "context": top_results}

    def hybrid_rag(self, query: str, top_k: int = 3, model: str = "gemini") -> Dict:
        keyword_filtered = [
            chunk
            for chunk in self._all_chunks
            if any(keyword in chunk.page_content.lower() for keyword in query.lower().split())
        ]
        candidate_db = LangChainVectorDB()
        candidate_db.upsert_documents(keyword_filtered or self._all_chunks)
        top_results = get_top_k_results(candidate_db, query=query, top_k=top_k)

        retriever = candidate_db.as_retriever(top_k=top_k)
        if retriever is None:
            return {"strategy": "hybrid_rag", "answer": "No context found.", "context": []}

        chain = RetrievalQA.from_chain_type(
            llm=self.llm.get_model(model),
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": self.prompter.multishot_prompting_template()},
            return_source_documents=False,
        )
        answer = chain.invoke({"query": query})["result"]
        return {"strategy": "hybrid_rag", "answer": answer, "context": top_results}

    def graph_rag(self, query: str, top_k: int = 3, model: str = "llama2") -> Dict:
        # Lightweight graph: group chunks by type and route by intent.
        intent = self._detect_intent(query)
        selected = [chunk for chunk in self._all_chunks if chunk.metadata.get("type") == intent]
        candidate_db = LangChainVectorDB()
        candidate_db.upsert_documents(selected or self._all_chunks)
        top_results = get_top_k_results(candidate_db, query=query, top_k=top_k)

        retriever = candidate_db.as_retriever(top_k=top_k)
        if retriever is None:
            return {"strategy": "graph_rag", "intent": intent, "answer": "No context found.", "context": []}

        chain = RetrievalQA.from_chain_type(
            llm=self.llm.get_model(model),
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": self.prompter.singleshot_prompting_template()},
            return_source_documents=False,
        )
        answer = chain.invoke({"query": query})["result"]
        return {"strategy": "graph_rag", "intent": intent, "answer": answer, "context": top_results}

    def agentic_rag(self, query: str, top_k: int = 4, model: str = "huggingface") -> Dict:
        """
        Simple agentic controller:
        1) detect intent,
        2) pick retrieval strategy,
        3) generate response.
        """
        intent = self._detect_intent(query)
        if intent == "promotions":
            retrieval = self.hybrid_rag(query, top_k=top_k, model=model)
        elif intent == "inventory":
            retrieval = self.graph_rag(query, top_k=top_k, model=model)
        else:
            retrieval = self.naive_rag(query, top_k=top_k, model=model)
        retrieval["strategy"] = "agentic_rag"
        retrieval["intent"] = intent
        return retrieval

    @staticmethod
    def _detect_intent(query: str) -> str:
        normalized = query.lower()
        if any(token in normalized for token in ["stock", "inventory", "available", "price"]):
            return "inventory"
        if any(token in normalized for token in ["deal", "discount", "offer", "promotion", "sale"]):
            return "promotions"
        return "general"
