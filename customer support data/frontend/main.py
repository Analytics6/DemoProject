import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from frontend.app_backend import authenticate_user, get_dashboard_stats, get_user_profile, initialize_database, list_tickets, log_audit_event
from augmentation.rag import RagModel

initialize_database()

st.set_page_config(page_title="Enterprise Retail Support", page_icon="🏪", layout="wide")

if "rag_model" not in st.session_state:
    st.session_state.rag_model = RagModel(data_dir=str(ROOT / "data"))

if "auth_user" not in st.session_state:
    st.session_state.auth_user = None

if "page" not in st.session_state:
    st.session_state.page = "login"


if st.session_state.page == "login":
    st.title("Enterprise Retail Support Platform")
    st.caption("Secure support operations, AI assistance, and customer engagement in one place.")

    with st.form("login_form"):
        username = st.text_input("Username", value="admin")
        password = st.text_input("Password", type="password", value="Admin@123!")
        submitted = st.form_submit_button("Sign In")

        if submitted:
            if authenticate_user(username, password):
                profile = get_user_profile(username)
                st.session_state.auth_user = profile
                st.session_state.page = "dashboard"
                log_audit_event(username, "login", "Successful sign in", db_path=str(ROOT / "databases" / "support_app.sqlite"))
                st.rerun()
            else:
                st.error("Invalid credentials")

    st.subheader("Sample credentials")
    st.code("admin / Admin@123!\nagent / Agent@123!\nanalyst / Analyst@123!")
    return


user = st.session_state.auth_user
if not user:
    st.session_state.page = "login"
    st.rerun()

st.sidebar.title("Operations Center")
st.sidebar.write(f"Signed in as {user['full_name']} ({user['role']})")
pages = ["dashboard", "tickets", "knowledge", "reports", "settings"]
selection = st.sidebar.radio("Navigate", pages, horizontal=False)
st.session_state.page = selection

if selection == "dashboard":
    st.title("Executive Dashboard")
    stats = get_dashboard_stats()
    cols = st.columns(3)
    cols[0].metric("Active Accounts", stats["active_users"])
    cols[1].metric("Open Tickets", stats["open_tickets"])
    cols[2].metric("Audit Events", stats["audit_events"])

    st.subheader("Recent activity")
    st.dataframe(list_tickets(), use_container_width=True)

elif selection == "tickets":
    st.title("Ticket Management")
    st.dataframe(list_tickets(), use_container_width=True)

elif selection == "knowledge":
    st.title("AI Knowledge Assistant")
    strategy = st.selectbox("RAG Strategy", options=["naive_rag", "hybrid_rag", "graph_rag", "agentic_rag"], index=3)
    llm_model = st.selectbox("LLM Model", options=["openai", "gemini", "llama2", "huggingface"], index=0)
    query = st.text_area("Ask the support assistant", placeholder="How do I handle a refund request?")
    if st.button("Generate answer") and query.strip():
        result = getattr(st.session_state.rag_model, strategy)(query, model=llm_model)
        st.write(result["answer"])
        st.caption("Retrieved context")
        for item in result.get("context", [])[:3]:
            st.write(item["chunk"]["text"])

elif selection == "reports":
    st.title("Operations Reports")
    st.info("Snapshot reporting and trend analysis are available for enterprise review.")

elif selection == "settings":
    st.title("Platform Settings")
    st.success("Authentication, audit logging, and deployment settings are configured.")
