from typing import Dict, List

from langchain_bootstrap import ensure_stdlib_logging

ensure_stdlib_logging()

from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory

class RedisMemory:
    """
    LangChain memory wrapper. In production, replace in-memory history
    with RedisChatMessageHistory by providing a Redis backend.
    """

    def __init__(self) -> None:
        self._store: Dict[str, ConversationBufferMemory] = {}

    def _get_memory(self, user_id: str) -> ConversationBufferMemory:
        memory = self._store.get(user_id)
        if memory is None:
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                chat_memory=ChatMessageHistory(),
            )
            self._store[user_id] = memory
        return memory

    def add_message(self, user_id: str, role: str, content: str) -> None:
        memory = self._get_memory(user_id)
        if role == "assistant":
            memory.chat_memory.add_ai_message(content)
        else:
            memory.chat_memory.add_user_message(content)

    def get_history(self, user_id: str, limit: int = 10) -> List[Dict]:
        memory = self._get_memory(user_id)
        messages = memory.chat_memory.messages[-limit:]
        output: List[Dict] = []
        for msg in messages:
            role = "assistant" if msg.type == "ai" else "user"
            output.append({"role": role, "content": msg.content})
        return output
