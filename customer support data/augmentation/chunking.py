from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(text: str, chunk_size: int = 220, overlap: int = 40) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap, separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_text(text)


def chunk_documents(
    documents: List[Document], chunk_size: int = 220, overlap: int = 40
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap, separators=["\n\n", "\n", ". ", " ", ""]
    )
    split_docs = splitter.split_documents(documents)

    for idx, doc in enumerate(split_docs):
        parent_id = doc.metadata.get("doc_id", f"DOC-{idx}")
        doc.metadata["chunk_id"] = f"{parent_id}-C{idx}"
    return split_docs
