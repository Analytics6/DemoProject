import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from augmentation.rag import RagModel  # noqa: E402


st.set_page_config(page_title="Retail Customer Support Chatbot", layout="wide")
st.title("Retail Customer Support Chatbot")
st.caption("Supports inventory, promotions, and general support queries.")

if "rag_model" not in st.session_state:
    st.session_state.rag_model = RagModel(data_dir=str(ROOT / "data"))

if "messages" not in st.session_state:
    st.session_state.messages = []

strategy = st.selectbox(
    "RAG Strategy",
    options=["naive_rag", "hybrid_rag", "graph_rag", "agentic_rag"],
    index=3,
)
llm_model = st.selectbox(
    "LLM Model",
    options=["openai", "gemini", "llama2", "huggingface"],
    index=0,
)

user_query = st.text_input(
    "Ask your question",
    placeholder="Example: Is Wireless Earbuds in stock? Any promotion on electronics?",
)

if st.button("Get Answer", type="primary") and user_query.strip():
    rag_model: RagModel = st.session_state.rag_model
    result = getattr(rag_model, strategy)(user_query, model=llm_model)

    st.session_state.messages.append({"role": "user", "content": user_query})
    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})

    st.subheader("Assistant Response")
    st.write(result["answer"])

    st.subheader("Retrieved Context (Top-K)")
    for item in result.get("context", []):
        st.markdown(f"**Score:** `{item['score']}`")
        st.write(item["chunk"]["text"])
        st.divider()

st.subheader("Chat History")
for msg in st.session_state.messages[-10:]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
