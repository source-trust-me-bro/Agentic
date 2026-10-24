"""Model-provider abstraction.

Agents never import a vendor SDK. They call ``LLM.generate`` with a *task name*, a
system prompt, a user prompt, the structured inputs (``payload``) and a Pydantic
``schema`` for the answer. Gemini uses the prompts and the schema; the offline
``FakeLLM`` uses the task name and the payload. Swapping providers touches only
this package.
"""

from __future__ import annotations

from typing import Any, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """The provider failed or returned something that does not match the schema."""


@runtime_checkable
class LLM(Protocol):
    name: str

    def generate(
        self,
        *,
        task: str,
        system: str,
        prompt: str,
        schema: type[T],
        payload: dict[str, Any] | None = None,
    ) -> T: ...
