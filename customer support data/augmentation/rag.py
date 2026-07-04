"""Backward-compatible RAG entrypoint — delegates to end-to-end pipeline."""
from services.rag_pipeline import EndToEndRAGPipeline, RagModel

__all__ = ["RagModel", "EndToEndRAGPipeline"]
