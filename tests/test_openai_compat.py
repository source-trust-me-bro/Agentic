"""The OpenAI-compatible provider (Hugging Face router), tested with a fake HTTP transport."""

from __future__ import annotations

import json

import httpx
import pytest
from pydantic import BaseModel

from newsroom.config import Settings
from newsroom.llm import build_llm
from newsroom.llm.base import LLMError
from newsroom.llm.openai_compat import OpenAICompatLLM, extract_json

TOKEN = "hf_test_secret_token_123"


class Out(BaseModel):
    answer: str


def reply(content: str, status: int = 200) -> httpx.Response:
    return httpx.Response(status, json={"choices": [{"message": {"content": content}}]})


def make(handler, retries: int = 2) -> tuple[OpenAICompatLLM, list[dict]]:
    seen: list[dict] = []

    def route(request: httpx.Request) -> httpx.Response:
        seen.append(
            {"auth": request.headers.get("authorization"), "body": json.loads(request.content)}
        )
        return handler(len(seen), request)

    llm = OpenAICompatLLM(
        api_key=TOKEN,
        model="deepseek-ai/test",
        base_url="https://router.example.test/v1",
        retries=retries,
        transport=httpx.MockTransport(route),
    )
    return llm, seen


def ask(llm: OpenAICompatLLM) -> Out:
    return llm.generate(task="t", system="be exact", prompt="question", schema=Out)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr("newsroom.llm.openai_compat.time.sleep", lambda _s: None)


def test_a_valid_answer_is_parsed_and_the_request_is_well_formed():
    llm, seen = make(lambda _n, _r: reply('{"answer": "yes"}'))

    assert ask(llm).answer == "yes"
    request = seen[0]
    assert request["auth"] == f"Bearer {TOKEN}"
    assert request["body"]["model"] == "deepseek-ai/test"
    assert request["body"]["response_format"] == {"type": "json_object"}
    assert "JSON schema" in request["body"]["messages"][0]["content"]  # the contract is stated
    assert request["body"]["messages"][1]["content"] == "question"


def test_reasoning_blocks_and_code_fences_are_stripped():
    wrapped = '<think>let me think about it</think>\n```json\n{"answer": "fenced"}\n```'
    llm, _ = make(lambda _n, _r: reply(wrapped))

    assert ask(llm).answer == "fenced"
    assert extract_json('noise {"a": 1} trailing') == '{"a": 1}'


def test_a_wrong_shape_is_asked_again_and_then_fails_cleanly():
    llm, seen = make(lambda _n, _r: reply('{"wrong": 1}'), retries=1)

    with pytest.raises(LLMError):
        ask(llm)
    assert len(seen) == 2  # one retry, with a corrective message
    assert "not valid JSON" in seen[1]["body"]["messages"][-1]["content"]


def test_a_bad_token_or_missing_model_fails_at_once_without_retrying():
    for status in (401, 403, 404, 402):
        llm, seen = make(lambda _n, _r, s=status: httpx.Response(s, text="refused"))
        with pytest.raises(LLMError):
            ask(llm)
        assert len(seen) == 1, status  # permanent errors are not retried


def test_rate_limits_and_server_errors_are_retried():
    def flaky(n, _r):
        return httpx.Response(429, text="slow down") if n < 3 else reply('{"answer": "ok"}')

    llm, seen = make(flaky, retries=2)

    assert ask(llm).answer == "ok"
    assert len(seen) == 3


def test_the_token_never_appears_in_an_error():
    llm, _ = make(lambda _n, _r: httpx.Response(401, text=f"bad key {TOKEN}"))

    with pytest.raises(LLMError) as error:
        ask(llm)
    assert TOKEN not in str(error.value)
    ok, message = llm.check()
    assert ok is False and TOKEN not in message


def test_check_reports_a_working_endpoint():
    llm, _ = make(lambda _n, _r: reply('{"ok": true}'))

    assert llm.check() == (True, "ok")


def test_the_provider_needs_a_token_and_is_wired_into_the_settings():
    with pytest.raises(LLMError):
        build_llm(Settings(llm_provider="huggingface", hf_token="", mongodb_uri=""))

    llm = build_llm(Settings(llm_provider="huggingface", hf_token=TOKEN, mongodb_uri=""))
    assert llm.name == "huggingface"
    assert Settings(mongodb_uri="").hf_model.startswith("deepseek-ai/")
