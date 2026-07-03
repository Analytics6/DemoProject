from typing import Dict, List

from augmentation.vectordb import LangChainVectorDB


def get_top_k_results(vector_db: LangChainVectorDB, query: str, top_k: int = 3) -> List[Dict]:
    ranked = vector_db.similarity_search(query=query, top_k=top_k)
    return [{"score": round(score, 4), "chunk": {"text": chunk.page_content, **chunk.metadata}} for score, chunk in ranked]
