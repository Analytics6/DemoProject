from pathlib import Path
from typing import Dict, List

from langchain_bootstrap import ensure_stdlib_logging

ensure_stdlib_logging()

from langchain_community.chat_message_histories import SQLChatMessageHistory


class SQLMemory:
    def __init__(self, db_path: str = "databases/chat_memory.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._histories: Dict[str, SQLChatMessageHistory] = {}

    def _get_history(self, user_id: str) -> SQLChatMessageHistory:
        history = self._histories.get(user_id)
        if history is None:
            history = SQLChatMessageHistory(
                session_id=user_id, connection=f"sqlite:///{self.db_path.as_posix()}"
            )
            self._histories[user_id] = history
        return history

    def add_message(self, user_id: str, role: str, content: str) -> None:
        history = self._get_history(user_id)
        if role == "assistant":
            history.add_ai_message(content)
        else:
            history.add_user_message(content)

    def get_history(self, user_id: str, limit: int = 10) -> List[Dict]:
        history = self._get_history(user_id)
        messages = history.messages[-limit:]
        output: List[Dict] = []
        for msg in messages:
            role = "assistant" if msg.type == "ai" else "user"
            output.append({"role": role, "content": msg.content})
        return output
