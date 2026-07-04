import os
import secrets
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain_bootstrap import ensure_stdlib_logging

ensure_stdlib_logging()

from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_message_histories import SQLChatMessageHistory

MEMORY_DB = ROOT / "databases" / "memory_store.sqlite"
CHAT_DB = ROOT / "databases" / "chat_memory.db"
SHORT_TERM_WINDOW = int(os.getenv("SHORT_TERM_MEMORY_WINDOW", "10"))


MEMORY_ARCHITECTURE_NOTES = {
    "title": "Session ID, Short-Term & Long-Term Memory — Full Notes",
    "overview": (
        "This support platform uses a three-layer memory architecture built entirely on LangChain "
        "and SQLite. Each AI chat is scoped to a Session ID. Short-term memory holds the active "
        "conversation window. Long-term memory persists user facts, preferences, and notes across "
        "sessions and days."
    ),
    "session_id": {
        "what": "A unique identifier (UUID) for each AI chat conversation.",
        "purpose": "Isolates conversation history so agents can run multiple parallel chats without cross-talk.",
        "storage": "SQLite table `ai_sessions` in databases/memory_store.sqlite",
        "lifecycle": "Created on 'New Session' → active while chatting → can be archived or deleted.",
        "langchain": "Used as `session_id` in SQLChatMessageHistory for short-term message storage.",
        "example": "sess_a8f3k2m9x1q7",
    },
    "short_term_memory": {
        "what": "The last N messages in the current session (default: 10 turns).",
        "purpose": "Gives the AI immediate conversational context — follow-ups like 'what about returns?' work correctly.",
        "storage": "LangChain SQLChatMessageHistory in databases/chat_memory.db",
        "langchain_class": "ConversationBufferWindowMemory (k=10)",
        "behavior": "Older messages roll off the window but remain in SQLite until session is cleared.",
        "injection": "Recent turns are prepended to the RAG query before retrieval and generation.",
    },
    "long_term_memory": {
        "what": "Persistent notes, facts, and preferences stored per user across all sessions.",
        "purpose": "Remembers customer preferences, recurring issues, and agent notes beyond a single chat.",
        "storage": "SQLite table `long_term_memory` in databases/memory_store.sqlite",
        "memory_types": ["note", "fact", "preference", "summary"],
        "behavior": "Loaded on every chat request and injected into the prompt context.",
        "langchain": "Custom store compatible with LangChain RunnableWithMessageHistory patterns.",
    },
    "data_flow": [
        "1. User opens AI page → active Session ID loaded or new session created.",
        "2. On each message → long-term memories fetched for user.",
        "3. Short-term window (last 10 messages) loaded for session_id.",
        "4. Context + query sent to LangChain RAG chain.",
        "5. Response saved to short-term memory (SQLChatMessageHistory).",
        "6. User can manually save any insight to long-term memory.",
    ],
    "database_schema": {
        "ai_sessions": "session_id, username, title, created_at, updated_at, is_active",
        "long_term_memory": "id, username, session_id, memory_key, memory_value, memory_type, created_at, updated_at",
        "message_store": "LangChain-managed tables in chat_memory.db (session_id → messages)",
    },
    "best_practices": [
        "Start a new session for each distinct customer inquiry.",
        "Save important resolutions to long-term memory for future reference.",
        "Use memory_type 'preference' for customer-specific settings.",
        "Use memory_type 'summary' for ticket resolution summaries.",
        "Clear short-term memory when switching to a unrelated topic in the same session.",
    ],
}


def _now() -> str:
    from datetime import datetime
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


def _conn(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_memory_database() -> None:
    conn = _conn(MEMORY_DB)
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS ai_sessions (
            session_id TEXT PRIMARY KEY,
            username TEXT NOT NULL,
            title TEXT NOT NULL DEFAULT 'New Chat',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            is_active INTEGER NOT NULL DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS long_term_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            session_id TEXT,
            memory_key TEXT NOT NULL,
            memory_value TEXT NOT NULL,
            memory_type TEXT NOT NULL DEFAULT 'note',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """
    )
    conn.commit()
    conn.close()
    CHAT_DB.parent.mkdir(parents=True, exist_ok=True)


class MemoryService:
    """Manages session IDs, LangChain short-term and SQLite long-term memory."""

    def __init__(self, short_term_window: int = SHORT_TERM_WINDOW) -> None:
        initialize_memory_database()
        self.short_term_window = short_term_window
        self._buffer_memories: Dict[str, ConversationBufferWindowMemory] = {}

    def create_session(self, username: str, title: str = "New Chat") -> Dict:
        session_id = f"sess_{secrets.token_urlsafe(12)}"
        now = _now()
        conn = _conn(MEMORY_DB)
        conn.execute(
            "INSERT INTO ai_sessions (session_id, username, title, created_at, updated_at, is_active) "
            "VALUES (?, ?, ?, ?, ?, 1)",
            (session_id, username, title, now, now),
        )
        conn.commit()
        conn.close()
        return {"session_id": session_id, "username": username, "title": title, "created_at": now}

    def list_sessions(self, username: str, active_only: bool = True) -> List[Dict]:
        conn = _conn(MEMORY_DB)
        query = "SELECT session_id, username, title, created_at, updated_at, is_active FROM ai_sessions WHERE username = ?"
        if active_only:
            query += " AND is_active = 1"
        query += " ORDER BY updated_at DESC"
        rows = conn.execute(query, (username,)).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_session(self, session_id: str, username: str) -> Optional[Dict]:
        conn = _conn(MEMORY_DB)
        row = conn.execute(
            "SELECT session_id, username, title, created_at, updated_at, is_active "
            "FROM ai_sessions WHERE session_id = ? AND username = ?",
            (session_id, username),
        ).fetchone()
        conn.close()
        return dict(row) if row else None

    def touch_session(self, session_id: str) -> None:
        conn = _conn(MEMORY_DB)
        conn.execute("UPDATE ai_sessions SET updated_at = ? WHERE session_id = ?", (_now(), session_id))
        conn.commit()
        conn.close()

    def archive_session(self, session_id: str, username: str) -> bool:
        conn = _conn(MEMORY_DB)
        cur = conn.execute(
            "UPDATE ai_sessions SET is_active = 0, updated_at = ? WHERE session_id = ? AND username = ?",
            (_now(), session_id, username),
        )
        conn.commit()
        conn.close()
        return cur.rowcount > 0

    def _get_buffer_memory(self, session_id: str) -> ConversationBufferWindowMemory:
        if session_id not in self._buffer_memories:
            history = SQLChatMessageHistory(
                session_id=session_id,
                connection=f"sqlite:///{CHAT_DB.as_posix()}",
            )
            self._buffer_memories[session_id] = ConversationBufferWindowMemory(
                k=self.short_term_window,
                chat_memory=history,
                return_messages=True,
                memory_key="chat_history",
            )
        return self._buffer_memories[session_id]

    def add_short_term(self, session_id: str, role: str, content: str) -> None:
        memory = self._get_buffer_memory(session_id)
        if role == "assistant":
            memory.chat_memory.add_ai_message(content)
        else:
            memory.chat_memory.add_user_message(content)
        self.touch_session(session_id)

    def get_short_term_history(self, session_id: str, limit: Optional[int] = None) -> List[Dict]:
        memory = self._get_buffer_memory(session_id)
        messages = memory.chat_memory.messages
        if limit:
            messages = messages[-limit:]
        output = []
        for msg in messages:
            role = "assistant" if msg.type == "ai" else "user"
            output.append({"role": role, "content": msg.content})
        return output

    def clear_short_term(self, session_id: str) -> None:
        memory = self._get_buffer_memory(session_id)
        memory.clear()
        if session_id in self._buffer_memories:
            del self._buffer_memories[session_id]

    def format_short_term_context(self, session_id: str) -> str:
        history = self.get_short_term_history(session_id)
        if not history:
            return ""
        lines = []
        for msg in history:
            label = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{label}: {msg['content']}")
        return "\n".join(lines)

    def add_long_term(
        self,
        username: str,
        memory_key: str,
        memory_value: str,
        memory_type: str = "note",
        session_id: Optional[str] = None,
    ) -> Dict:
        now = _now()
        conn = _conn(MEMORY_DB)
        cur = conn.execute(
            "INSERT INTO long_term_memory (username, session_id, memory_key, memory_value, memory_type, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (username, session_id, memory_key, memory_value, memory_type, now, now),
        )
        memory_id = cur.lastrowid
        conn.commit()
        conn.close()
        return {
            "id": memory_id,
            "username": username,
            "session_id": session_id,
            "memory_key": memory_key,
            "memory_value": memory_value,
            "memory_type": memory_type,
            "created_at": now,
        }

    def list_long_term(self, username: str, memory_type: Optional[str] = None) -> List[Dict]:
        conn = _conn(MEMORY_DB)
        query = (
            "SELECT id, username, session_id, memory_key, memory_value, memory_type, created_at, updated_at "
            "FROM long_term_memory WHERE username = ?"
        )
        params: list = [username]
        if memory_type:
            query += " AND memory_type = ?"
            params.append(memory_type)
        query += " ORDER BY updated_at DESC"
        rows = conn.execute(query, params).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def delete_long_term(self, memory_id: int, username: str) -> bool:
        conn = _conn(MEMORY_DB)
        cur = conn.execute(
            "DELETE FROM long_term_memory WHERE id = ? AND username = ?",
            (memory_id, username),
        )
        conn.commit()
        conn.close()
        return cur.rowcount > 0

    def format_long_term_context(self, username: str) -> str:
        memories = self.list_long_term(username)
        if not memories:
            return ""
        lines = []
        for m in memories[:20]:
            lines.append(f"[{m['memory_type']}] {m['memory_key']}: {m['memory_value']}")
        return "\n".join(lines)

    def build_contextual_query(self, session_id: str, username: str, message: str) -> str:
        short_ctx = self.format_short_term_context(session_id)
        long_ctx = self.format_long_term_context(username)
        parts = []
        if long_ctx:
            parts.append(f"Long-term memory (persistent user context):\n{long_ctx}")
        if short_ctx:
            parts.append(f"Short-term memory (current session, last {self.short_term_window} turns):\n{short_ctx}")
        parts.append(f"Current user question: {message}")
        return "\n\n".join(parts)

    def get_memory_notes(self) -> Dict:
        return MEMORY_ARCHITECTURE_NOTES

    def get_memory_status(self, session_id: str, username: str) -> Dict:
        session = self.get_session(session_id, username)
        short = self.get_short_term_history(session_id)
        long_term = self.list_long_term(username)
        return {
            "session": session,
            "session_id": session_id,
            "short_term_count": len(short),
            "short_term_window": self.short_term_window,
            "long_term_count": len(long_term),
            "short_term_preview": short[-3:],
        }
