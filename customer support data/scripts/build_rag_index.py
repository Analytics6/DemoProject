"""Build the end-to-end RAG FAISS index from all data sources."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from scripts.generate_tickets import generate_tickets
from services.rag_pipeline import EndToEndRAGPipeline
import json


def main() -> None:
    tickets_path = ROOT / "data" / "tickets.json"
    if not tickets_path.exists():
        tickets_path.write_text(json.dumps(generate_tickets(500), indent=2), encoding="utf-8")
        print(f"Generated 500 tickets → {tickets_path}")

    print("Building end-to-end RAG index…")
    pipeline = EndToEndRAGPipeline(data_dir=str(ROOT / "data"))
    stats = pipeline.build_index()
    status = pipeline.get_pipeline_status()

    print(f"  Source docs:  {stats.total_source_docs}")
    print(f"  Ticket docs:  {stats.ticket_docs}")
    print(f"  Total chunks: {stats.total_chunks}")
    print(f"  Embedding:    {stats.embedding_model}")
    print(f"  Index path:   {stats.index_path}")
    print("Pipeline stages:", [s["label"] for s in status["stages"]])
    print("Done.")


if __name__ == "__main__":
    main()
