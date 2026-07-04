import os
from typing import Any, List, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import LLM
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class EchoRetailLLM(LLM):
    provider_name: str = "openai"

    @property
    def _llm_type(self) -> str:
        return f"echo_{self.provider_name}"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop:
            for token in stop:
                if token in prompt:
                    prompt = prompt.split(token)[0]
        trimmed = prompt if len(prompt) <= 500 else f"{prompt[:500]}..."
        return f"[{self.provider_name} via LangChain — set OPENAI_API_KEY for live responses]\n{trimmed}"


class EchoChatModel(BaseChatModel):
    provider_name: str = "openai"
    model_name: str = "echo"

    @property
    def _llm_type(self) -> str:
        return f"echo_chat_{self.provider_name}"

    def _generate(
        self,
        messages: List,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        text_parts = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                text_parts.append(f"System: {msg.content}")
            elif isinstance(msg, HumanMessage):
                text_parts.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                text_parts.append(f"Assistant: {msg.content}")
            else:
                text_parts.append(str(msg.content))
        prompt = "\n".join(text_parts)
        response = EchoRetailLLM(provider_name=self.provider_name)._call(prompt, stop=stop)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=response))])


class LLMModels:
    """LangChain LLM adapters with OpenAI support when API key is configured."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

    def get_model(self, provider: str = "openai"):
        selected = provider.lower()
        if selected == "openai" and self.api_key:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                api_key=self.api_key,
                temperature=0.2,
            )
        if selected == "openai":
            return EchoChatModel(provider_name="openai")
        if selected == "gemini":
            return EchoChatModel(provider_name="gemini")
        if selected == "llama2":
            return EchoChatModel(provider_name="llama2")
        if selected in {"huggingface", "hugging_face", "hf"}:
            return EchoChatModel(provider_name="huggingface")
        return EchoChatModel(provider_name=selected)

    def get_llm(self, provider: str = "openai") -> LLM:
        """Return a text LLM for legacy RetrievalQA chains."""
        model = self.get_model(provider)
        if isinstance(model, LLM):
            return model
        from langchain_openai import OpenAI

        if provider == "openai" and self.api_key:
            return OpenAI(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                api_key=self.api_key,
                temperature=0.2,
            )
        return EchoRetailLLM(provider_name=provider)

    def openai(self, prompt: str) -> str:
        return self.get_model("openai").invoke(prompt).content

    def gemini(self, prompt: str) -> str:
        return self.get_model("gemini").invoke(prompt).content

    def llama2(self, prompt: str) -> str:
        return self.get_model("llama2").invoke(prompt).content

    def hugging_face_models(self, prompt: str) -> str:
        return self.get_model("huggingface").invoke(prompt).content
