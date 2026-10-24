from __future__ import annotations

import importlib
import pkgutil

from newsroom.config import Settings, get_settings
from newsroom.llm.base import LLM, LLMError
from newsroom.llm.fake import FakeLLM, Handler

__all__ = ["LLM", "LLMError", "FakeLLM", "build_llm", "collect_fake_handlers"]


def collect_fake_handlers() -> dict[str, Handler]:
    """Gather ``FAKE_HANDLERS`` from every agent module."""

    import newsroom.agents as agents_pkg

    handlers: dict[str, Handler] = {}
    for module_info in pkgutil.iter_modules(agents_pkg.__path__):
        module = importlib.import_module(f"newsroom.agents.{module_info.name}")
        handlers.update(getattr(module, "FAKE_HANDLERS", {}))
    return handlers


def build_llm(settings: Settings | None = None) -> LLM:
    settings = settings or get_settings()
    if settings.llm_provider == "gemini":
        from newsroom.llm.gemini import GeminiLLM

        return GeminiLLM(
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
            timeout_seconds=settings.llm_timeout_seconds,
            thinking_level=settings.gemini_thinking_level,
        )
    if settings.llm_provider == "huggingface":
        from newsroom.llm.openai_compat import OpenAICompatLLM

        return OpenAICompatLLM(
            name="huggingface",
            api_key=settings.hf_token,
            model=settings.hf_model,
            base_url=settings.hf_base_url,
            timeout_seconds=settings.llm_timeout_seconds,
        )
    return FakeLLM(collect_fake_handlers())
