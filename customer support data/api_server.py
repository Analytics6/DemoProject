import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from frontend.app_backend import (  # noqa: E402
    bulk_load_tickets_from_json,
    create_session,
    create_ticket,
    get_analytics,
    get_dashboard_stats,
    get_user_profile,
    initialize_database,
    list_audit_log,
    list_customers,
    list_integrations,
    list_tickets,
    list_users,
    log_audit_event,
    revoke_session,
    update_ticket_status,
    validate_session,
)
from frontend.app_backend import authenticate_user  # noqa: E402
from frontend.auth_service import get_auth_status, list_login_history  # noqa: E402
from services.ai_service import AIService  # noqa: E402

DB_PATH = str(ROOT / "databases" / "support_app.sqlite")
initialize_database(db_path=DB_PATH)

# Generate 500 tickets + load into SQLite for RAG training data
_tickets_json = ROOT / "data" / "tickets.json"
if not _tickets_json.exists():
    from scripts.generate_tickets import generate_tickets
    import json as _json
    _tickets_json.parent.mkdir(parents=True, exist_ok=True)
    _tickets_json.write_text(_json.dumps(generate_tickets(500), indent=2), encoding="utf-8")
bulk_load_tickets_from_json(str(_tickets_json), db_path=DB_PATH)

ai_service = AIService(
    data_dir=str(ROOT / "data"),
    api_key=os.getenv("OPENAI_API_KEY"),
)

app = FastAPI(
    title="Enterprise Retail Support API",
    description="Production API with SQLite, LangChain RAG, and OpenAI",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LoginRequest(BaseModel):
    username: str
    password: str


class TicketCreateRequest(BaseModel):
    customer_name: str
    subject: str
    priority: str = "medium"
    assigned_to: str = "agent"


class TicketUpdateRequest(BaseModel):
    status: str


class ChatRequest(BaseModel):
    message: str
    strategy: str = "agentic_rag"
    model: str = "openai"
    session_id: Optional[str] = None


class LongTermMemoryRequest(BaseModel):
    memory_key: str
    memory_value: str
    memory_type: str = "note"
    session_id: Optional[str] = None


class SessionCreateRequest(BaseModel):
    title: str = "New Chat"


class RagRequest(BaseModel):
    query: str
    strategy: str = "agentic_rag"
    model: str = "openai"
    top_k: int = 4


class TicketSummaryRequest(BaseModel):
    subject: str
    customer_name: str
    model: str = "openai"


def get_current_user(authorization: Optional[str] = Header(default=None)) -> str:
    token = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization[7:]
    username = validate_session(token, db_path=DB_PATH) if token else None
    if not username:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return username


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "openai_configured": ai_service.openai_configured,
        "auth": get_auth_status(db_path=DB_PATH),
        "vector_store": "chromadb",
        "stats": get_dashboard_stats(db_path=DB_PATH),
        "rag": ai_service.get_pipeline_status()["stats"],
    }


@app.post("/auth/login")
def login(payload: LoginRequest, request: Request):
    client = request.client.host if request.client else ""
    ua = request.headers.get("user-agent", "")[:200]
    if not authenticate_user(
        payload.username, payload.password, db_path=DB_PATH,
        ip_address=client, user_agent=ua,
    ):
        return JSONResponse(status_code=401, content={"error": "Invalid credentials"})
    token = create_session(payload.username, db_path=DB_PATH)
    profile = get_user_profile(payload.username, db_path=DB_PATH)
    log_audit_event(payload.username, "login", "Successful sign in", db_path=DB_PATH)
    return {
        "token": token,
        "user": profile,
        "stats": get_dashboard_stats(db_path=DB_PATH),
        "auth": get_auth_status(db_path=DB_PATH),
    }


@app.get("/auth/status")
def auth_status(username: str = Depends(get_current_user)):
    return {
        "auth": get_auth_status(db_path=DB_PATH),
        "recent_logins": list_login_history(limit=10, db_path=DB_PATH),
    }


@app.post("/auth")
def auth_legacy(payload: dict):
    """Backward-compatible login endpoint."""
    username = payload.get("username", "")
    password = payload.get("password", "")
    if not authenticate_user(username, password, db_path=DB_PATH):
        return JSONResponse(status_code=401, content={"error": "Invalid credentials"})
    profile = get_user_profile(username, db_path=DB_PATH)
    log_audit_event(username, "login", "Successful sign in", db_path=DB_PATH)
    return {
        "user": profile,
        "stats": get_dashboard_stats(db_path=DB_PATH),
        "tickets": list_tickets(db_path=DB_PATH, per_page=10000)["tickets"],
    }


@app.post("/auth/logout")
def logout(authorization: Optional[str] = Header(default=None)):
    if authorization and authorization.startswith("Bearer "):
        revoke_session(authorization[7:], db_path=DB_PATH)
    return {"status": "logged_out"}


@app.get("/me")
def me(username: str = Depends(get_current_user)):
    return get_user_profile(username, db_path=DB_PATH)


@app.get("/stats")
def stats(username: str = Depends(get_current_user)):
    return get_dashboard_stats(db_path=DB_PATH)


@app.get("/tickets")
def tickets(
    page: int = 1,
    per_page: int = 50,
    status: Optional[str] = None,
    category: Optional[str] = None,
    search: Optional[str] = None,
    username: str = Depends(get_current_user),
):
    return list_tickets(
        db_path=DB_PATH,
        page=page,
        per_page=per_page,
        status=status,
        category=category,
        search=search,
    )


@app.post("/tickets")
def add_ticket(payload: TicketCreateRequest, username: str = Depends(get_current_user)):
    ticket = create_ticket(
        customer_name=payload.customer_name,
        subject=payload.subject,
        priority=payload.priority,
        assigned_to=payload.assigned_to,
        db_path=DB_PATH,
    )
    log_audit_event(username, "ticket_create", f"Created ticket #{ticket['id']}", db_path=DB_PATH)
    return ticket


@app.patch("/tickets/{ticket_id}")
def patch_ticket(
    ticket_id: int,
    payload: TicketUpdateRequest,
    username: str = Depends(get_current_user),
):
    ticket = update_ticket_status(ticket_id, payload.status, db_path=DB_PATH)
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    log_audit_event(
        username,
        "ticket_update",
        f"Updated ticket #{ticket_id} to {payload.status}",
        db_path=DB_PATH,
    )
    return ticket


@app.get("/customers")
def customers(username: str = Depends(get_current_user)):
    return {"customers": list_customers(db_path=DB_PATH)}


@app.get("/analytics")
def analytics(username: str = Depends(get_current_user)):
    return {
        "stats": get_dashboard_stats(db_path=DB_PATH),
        "analytics": get_analytics(db_path=DB_PATH),
    }


@app.get("/audit-log")
def audit_log(username: str = Depends(get_current_user)):
    return {"events": list_audit_log(db_path=DB_PATH)}


@app.get("/integrations")
def integrations(username: str = Depends(get_current_user)):
    return {"integrations": list_integrations(db_path=DB_PATH)}


@app.get("/admin/users")
def admin_users(username: str = Depends(get_current_user)):
    profile = get_user_profile(username, db_path=DB_PATH)
    if profile and profile.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return {"users": list_users(db_path=DB_PATH)}


@app.get("/ai/config")
def ai_config(username: str = Depends(get_current_user)):
    pipeline = ai_service.get_pipeline_status()
    return {
        "strategies": AIService.STRATEGIES,
        "models": AIService.MODELS,
        "openai_configured": ai_service.openai_configured,
        "pipeline": pipeline,
    }


@app.get("/rag/status")
def rag_status(username: str = Depends(get_current_user)):
    return ai_service.get_pipeline_status()


@app.post("/rag/rebuild")
def rag_rebuild(username: str = Depends(get_current_user)):
    profile = get_user_profile(username, db_path=DB_PATH)
    if profile and profile.get("role") not in ("admin", "analyst"):
        raise HTTPException(status_code=403, detail="Admin or analyst role required")
    result = ai_service.rebuild_index()
    log_audit_event(username, "rag_rebuild", "Rebuilt FAISS index", db_path=DB_PATH)
    return result


@app.post("/rag/query")
def rag_query(payload: RagRequest, username: str = Depends(get_current_user)):
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query is required")
    result = ai_service.run_rag(
        query=payload.query.strip(),
        strategy=payload.strategy,
        model=payload.model,
        top_k=payload.top_k,
    )
    log_audit_event(username, "rag_query", payload.query[:120], db_path=DB_PATH)
    return result


@app.post("/ai/chat")
def ai_chat(payload: ChatRequest, username: str = Depends(get_current_user)):
    if not payload.message.strip():
        raise HTTPException(status_code=400, detail="Message is required")
    result = ai_service.chat(
        user_id=username,
        message=payload.message.strip(),
        session_id=payload.session_id,
        strategy=payload.strategy,
        model=payload.model,
    )
    log_audit_event(
        username,
        "ai_chat",
        f"[{result['session_id']}] {payload.message[:100]}",
        db_path=DB_PATH,
    )
    return result


@app.post("/ai/sessions")
def create_ai_session(payload: SessionCreateRequest, username: str = Depends(get_current_user)):
    session = ai_service.memory.create_session(username, title=payload.title)
    log_audit_event(username, "ai_session_create", session["session_id"], db_path=DB_PATH)
    return session


@app.get("/ai/sessions")
def list_ai_sessions(username: str = Depends(get_current_user)):
    return {"sessions": ai_service.memory.list_sessions(username)}


@app.get("/ai/sessions/{session_id}")
def get_ai_session(session_id: str, username: str = Depends(get_current_user)):
    session = ai_service.memory.get_session(session_id, username)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    status = ai_service.memory.get_memory_status(session_id, username)
    return {"session": session, "memory_status": status}


@app.delete("/ai/sessions/{session_id}")
def archive_ai_session(session_id: str, username: str = Depends(get_current_user)):
    if not ai_service.memory.archive_session(session_id, username):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "archived", "session_id": session_id}


@app.get("/ai/memory/short/{session_id}")
def get_short_term_memory(session_id: str, username: str = Depends(get_current_user)):
    if not ai_service.memory.get_session(session_id, username):
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": session_id,
        "window_size": ai_service.memory.short_term_window,
        "messages": ai_service.memory.get_short_term_history(session_id),
    }


@app.delete("/ai/memory/short/{session_id}")
def clear_short_term_memory(session_id: str, username: str = Depends(get_current_user)):
    if not ai_service.memory.get_session(session_id, username):
        raise HTTPException(status_code=404, detail="Session not found")
    ai_service.memory.clear_short_term(session_id)
    return {"status": "cleared", "session_id": session_id}


@app.get("/ai/memory/long")
def get_long_term_memory(username: str = Depends(get_current_user), memory_type: Optional[str] = None):
    return {"memories": ai_service.memory.list_long_term(username, memory_type=memory_type)}


@app.post("/ai/memory/long")
def add_long_term_memory(payload: LongTermMemoryRequest, username: str = Depends(get_current_user)):
    if not payload.memory_key.strip() or not payload.memory_value.strip():
        raise HTTPException(status_code=400, detail="memory_key and memory_value are required")
    memory = ai_service.memory.add_long_term(
        username=username,
        memory_key=payload.memory_key.strip(),
        memory_value=payload.memory_value.strip(),
        memory_type=payload.memory_type,
        session_id=payload.session_id,
    )
    log_audit_event(username, "long_term_memory_add", payload.memory_key[:80], db_path=DB_PATH)
    return memory


@app.delete("/ai/memory/long/{memory_id}")
def delete_long_term_memory(memory_id: int, username: str = Depends(get_current_user)):
    if not ai_service.memory.delete_long_term(memory_id, username):
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"status": "deleted", "id": memory_id}


@app.get("/ai/memory/notes")
def get_memory_notes(username: str = Depends(get_current_user)):
    return ai_service.memory.get_memory_notes()


@app.get("/ai/memory/status")
def get_memory_status(session_id: str, username: str = Depends(get_current_user)):
    if not ai_service.memory.get_session(session_id, username):
        raise HTTPException(status_code=404, detail="Session not found")
    return ai_service.memory.get_memory_status(session_id, username)


@app.post("/ai/rag")
def ai_rag(payload: RagRequest, username: str = Depends(get_current_user)):
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query is required")
    result = ai_service.run_rag(
        query=payload.query.strip(),
        strategy=payload.strategy,
        model=payload.model,
        top_k=payload.top_k,
    )
    log_audit_event(username, "ai_rag", payload.query[:120], db_path=DB_PATH)
    return result


@app.get("/ai/history")
def ai_history(session_id: Optional[str] = None, username: str = Depends(get_current_user)):
    return {
        "history": ai_service.get_chat_history(username, session_id=session_id),
        "session_id": session_id,
    }


@app.post("/ai/summarize-ticket")
def summarize_ticket(payload: TicketSummaryRequest, username: str = Depends(get_current_user)):
    summary = ai_service.summarize_ticket(
        subject=payload.subject,
        customer_name=payload.customer_name,
        model=payload.model,
    )
    return {"summary": summary}
