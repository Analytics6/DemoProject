import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain_bootstrap import ensure_stdlib_logging

ensure_stdlib_logging()

from generation.llm import LLMModels
from services.memory_service import MemoryService
from services.rag_pipeline import EndToEndRAGPipeline


class AIService:
    """Production LangChain service with end-to-end RAG and session memory."""

    STRATEGIES = EndToEndRAGPipeline.STRATEGIES
    MODELS = ["openai", "gemini", "llama2", "huggingface"]

    def __init__(self, data_dir: Optional[str] = None, api_key: Optional[str] = None) -> None:
        self.data_dir = data_dir or str(ROOT / "data")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.llm_models = LLMModels(api_key=self.api_key)
        self.rag = EndToEndRAGPipeline(data_dir=self.data_dir, api_key=self.api_key)
        self.rag.llm = self.llm_models
        self.memory = MemoryService()

    @property
    def openai_configured(self) -> bool:
        return bool(self.api_key)

    def get_pipeline_status(self) -> Dict:
        return self.rag.get_pipeline_status()

    def rebuild_index(self) -> Dict:
        stats = self.rag.rebuild_index()
        return {"status": "rebuilt", "stats": self.rag.get_pipeline_status()["stats"]}

    def ensure_session(self, username: str, session_id: Optional[str] = None) -> str:
        if session_id:
            session = self.memory.get_session(session_id, username)
            if session:
                return session_id
        created = self.memory.create_session(username)
        return created["session_id"]

    def run_rag(
        self,
        query: str,
        strategy: str = "agentic_rag",
        model: str = "openai",
        top_k: int = 4,
    ) -> Dict:
        if strategy not in self.STRATEGIES:
            strategy = "agentic_rag"
        result = self.rag.query(query, strategy=strategy, model=model, top_k=top_k)
        result["openai_configured"] = self.openai_configured
        result["pipeline_stats"] = self.rag.get_pipeline_status()["stats"]
        return result

    def chat(
        self,
        user_id: str,
        message: str,
        session_id: Optional[str] = None,
        strategy: str = "agentic_rag",
        model: str = "openai",
    ) -> Dict:
        active_session = self.ensure_session(user_id, session_id)
        contextual_query = self.memory.build_contextual_query(active_session, user_id, message)
        self.memory.add_short_term(active_session, "user", message)
        result = self.run_rag(contextual_query, strategy=strategy, model=model)
        answer = result.get("answer", "I could not generate a response.")
        self.memory.add_short_term(active_session, "assistant", answer)
        history = self.memory.get_short_term_history(active_session)
        long_term = self.memory.list_long_term(user_id)
        status = self.memory.get_memory_status(active_session, user_id)
        return {
            "answer": answer,
            "session_id": active_session,
            "strategy": result.get("strategy", strategy),
            "intent": result.get("intent"),
            "context": result.get("context", [])[:3],
            "history": history,
            "short_term_memory": history,
            "long_term_memory": long_term,
            "memory_status": status,
            "pipeline_stats": result.get("pipeline_stats"),
            "openai_configured": self.openai_configured,
        }

    def get_chat_history(self, user_id: str, session_id: Optional[str] = None, limit: int = 20) -> List[Dict]:
        if not session_id:
            sessions = self.memory.list_sessions(user_id)
            if not sessions:
                return []
            session_id = sessions[0]["session_id"]
        return self.memory.get_short_term_history(session_id, limit=limit)

    def summarize_ticket(self, subject: str, customer_name: str, model: str = "openai") -> str:
        from langchain.chains.llm import LLMChain
        from langchain_core.prompts import PromptTemplate

        prompt = PromptTemplate(
            input_variables=["subject", "customer"],
            template=(
                "You are a retail support analyst. Summarize this support ticket in 2-3 sentences "
                "and suggest the next best action.\n\n"
                "Customer: {customer}\n"
                "Subject: {subject}\n\n"
                "Summary and next action:"
            ),
        )
        chain = LLMChain(llm=self.llm_models.get_llm(model), prompt=prompt)
        return chain.run(subject=subject, customer=customer_name)
