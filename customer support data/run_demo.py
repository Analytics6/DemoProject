"""Demo: run agentic RAG against the 500-ticket knowledge base."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.rag_pipeline import EndToEndRAGPipeline


def run_demo() -> None:
    pipeline = EndToEndRAGPipeline(data_dir=str(ROOT / "data"))
    status = pipeline.get_pipeline_status()
    stats = status["stats"]
    print(f"RAG index: {stats['total_chunks']} chunks from {stats['total_source_docs']} docs ({stats['ticket_docs']} tickets)")

    sample_questions = [
        "How do I handle a refund request for order #1042?",
        "Is Wireless Earbuds available in stock?",
        "Do you have any promotions on apparel?",
        "What is your return policy?",
    ]

    for question in sample_questions:
        result = pipeline.agentic_rag(question, model="openai")
        print("=" * 80)
        print(f"Question: {question}")
        print(f"Intent: {result.get('intent')}")
        print(f"Answer:\n{result['answer']}")


if __name__ == "__main__":
    run_demo()
