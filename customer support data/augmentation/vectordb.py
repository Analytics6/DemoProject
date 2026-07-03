from typing import List, Tuple

from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


class LangChainVectorDB:
    def __init__(self, embedding_size: int = 128) -> None:
        self._embeddings = FakeEmbeddings(size=embedding_size)
        self._vectorstore: FAISS | None = None

    def upsert_documents(self, documents: List[Document]) -> None:
        if not documents:
            self._vectorstore = None
            return

        self._vectorstore = FAISS.from_documents(documents=documents, embedding=self._embeddings)

    def similarity_search(self, query: str, top_k: int = 3) -> List[Tuple[float, Document]]:
        if self._vectorstore is None:
            return []
        raw = self._vectorstore.similarity_search_with_score(query, k=top_k)
        # FAISS score is distance (lower is better). Convert to pseudo similarity.
        return [(1.0 / (1.0 + score), doc) for doc, score in raw]

    def as_retriever(self, top_k: int = 3):
        if self._vectorstore is None:
            return None
        return self._vectorstore.as_retriever(search_kwargs={"k": top_k})
