import os
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHROMA_DIR = ROOT / "databases" / "chroma_db"
DEFAULT_COLLECTION = "retail_support_knowledge"


class LangChainVectorDB:
    """LangChain ChromaDB vector store with optional persistence."""

    def __init__(
        self,
        embedding_size: int = 128,
        api_key: Optional[str] = None,
        persist_directory: Optional[str] = None,
        collection_name: str = DEFAULT_COLLECTION,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self._embeddings = self._build_embeddings(embedding_size)
        self._vectorstore: Optional[Chroma] = None

    @property
    def embedding_model_name(self) -> str:
        if self.api_key:
            return "text-embedding-3-small"
        return "fake-embeddings"

    @property
    def vector_store_name(self) -> str:
        return "chromadb"

    def _build_embeddings(self, size: int):
        if self.api_key:
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(api_key=self.api_key, model="text-embedding-3-small")
        return FakeEmbeddings(size=size)

    def upsert_documents(self, documents: List[Document]) -> None:
        if not documents:
            self._vectorstore = None
            return
        kwargs = {
            "documents": documents,
            "embedding": self._embeddings,
            "collection_name": self.collection_name,
        }
        if self.persist_directory:
            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
            kwargs["persist_directory"] = self.persist_directory
        self._vectorstore = Chroma.from_documents(**kwargs)

    def save(self, directory: str) -> None:
        """Chroma persists automatically when persist_directory is set."""
        if self.persist_directory and self._vectorstore:
            try:
                self._vectorstore.persist()
            except Exception:
                pass

    def load(self, directory: str) -> None:
        path = Path(directory)
        if not path.exists():
            raise FileNotFoundError(f"ChromaDB path not found: {directory}")
        self.persist_directory = str(path)
        self._vectorstore = Chroma(
            persist_directory=str(path),
            embedding_function=self._embeddings,
            collection_name=self.collection_name,
        )

    @staticmethod
    def clear_persist_directory(directory: str) -> None:
        path = Path(directory)
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True, exist_ok=True)

    def similarity_search(self, query: str, top_k: int = 3) -> List[Tuple[float, Document]]:
        if self._vectorstore is None:
            return []
        raw = self._vectorstore.similarity_search_with_score(query, k=top_k)
        return [(1.0 / (1.0 + score), doc) for doc, score in raw]

    def as_retriever(self, top_k: int = 3):
        if self._vectorstore is None:
            return None
        return self._vectorstore.as_retriever(search_kwargs={"k": top_k})

    def document_count(self) -> int:
        if self._vectorstore is None:
            return 0
        try:
            return self._vectorstore._collection.count()
        except Exception:
            return 0
