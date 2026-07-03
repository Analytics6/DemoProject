# Customer Support Chatbot (Retail Client)

Gen AI chatbot supporting:
- Inventory questions
- Promotions questions
- General questions

## Project Structure

- `frontend/streamlit.py` - Streamlit chatbot UI
- `frontend/index.html` - static project landing page
- `retrieval/Preprocessing.py` - data preprocessing
- `retrieval/Cleantext.py` - text cleaning utility
- `augmentation/Fulltext.py` - full document creation
- `augmentation/chunking.py` - text chunking
- `augmentation/vectordb.py` - simple vector store
- `augmentation/topkresults.py` - top-k retrieval
- `augmentation/rag.py` - `RagModel` with:
  - `naive_rag`
  - `hybrid_rag`
  - `graph_rag`
  - `agentic_rag`
- `augmentation/memory-redis.py` - Redis-style memory (mock)
- `augmentation/memory-sql.py` - SQL memory (SQLite)
- `generation/prompt.py` - prompting class
- `generation/llm.py` - model adapters (mock)
- `generation/metrics.py` - BLEU, ROUGE, METEOR, perplexity, RAG/retrieval/generation metrics, human-in-the-loop helper
- `generation/modelfinetune.py` - LoRA, QLoRA, PEFT, RLHF, adapter merge quality strategy scaffolds
- `databases/users.sql` - users schema + sample data
- `logging/system_logs.py` - cloud/tools metrics logging
- `logging/application_logs.py` - app metrics logging
- `data/*` - sample dataset

## Run

```bash
pip install -r requirements.txt
python run_demo.py
streamlit run frontend/streamlit.py
```
