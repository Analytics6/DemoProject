from typing import Any, List, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM


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
        return f"[{self.provider_name} via LangChain]\n{trimmed}"


class LLMModels:
    """LangChain LLM adapters (mockable and chain-compatible)."""

    def get_model(self, provider: str = "openai") -> LLM:
        selected = provider.lower()
        if selected == "gemini":
            return EchoRetailLLM(provider_name="gemini")
        if selected == "llama2":
            return EchoRetailLLM(provider_name="llama2")
        if selected in {"huggingface", "hugging_face", "hf"}:
            return EchoRetailLLM(provider_name="huggingface")
        return EchoRetailLLM(provider_name="openai")

    def openai(self, prompt: str) -> str:
        return self.get_model("openai").invoke(prompt)

    def gemini(self, prompt: str) -> str:
        return self.get_model("gemini").invoke(prompt)

    def llama2(self, prompt: str) -> str:
        return self.get_model("llama2").invoke(prompt)

    def hugging_face_models(self, prompt: str) -> str:
        return self.get_model("huggingface").invoke(prompt)
