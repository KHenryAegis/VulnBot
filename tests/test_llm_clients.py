import json
from types import SimpleNamespace

import httpx
import openai
import pytest
from openai import OpenAI

from server.chat.errors import (LLMContentFiltered, LLMEmptyReply, LLMRequestFailed, LLMTruncated,
                                validate_completion)
from server.chat.llm_clients import OpenAIChat


def completion(content="ok", finish_reason="stop"):
    return httpx.Response(200, json={
        "id": "chatcmpl-1", "object": "chat.completion", "created": 0, "model": "m",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                     "finish_reason": finish_reason}],
    })


def make_chat(handler, **config):
    """An OpenAIChat whose HTTP layer is ``handler``; ``handler.calls`` counts requests."""
    calls = []

    def transport(request):
        calls.append(json.loads(request.content))
        return handler(len(calls), request)

    cfg = SimpleNamespace(api_key="k", base_url="http://llm.test/v1", timeout=5, llm_model_name="m",
                          temperature=0.5, max_tokens=0, max_retries=3, **config)
    client = OpenAI(api_key="k", base_url=cfg.base_url, max_retries=0,
                    http_client=httpx.Client(transport=httpx.MockTransport(transport)))
    chat = OpenAIChat(cfg, client=client)
    chat.calls = calls
    return chat


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(OpenAIChat, "sleep", staticmethod(lambda seconds: None))


HISTORY = [{"role": "user", "content": "hi"}]


def test_returns_reply():
    chat = make_chat(lambda n, r: completion("hello"))
    assert chat.chat(HISTORY) == "hello"
    assert len(chat.calls) == 1


@pytest.mark.parametrize("status", [429, 500, 502, 503])
def test_retries_transient_status_then_succeeds(status):
    chat = make_chat(lambda n, r: httpx.Response(status, json={"error": {"message": "busy"}}) if n < 3 else completion("ok"))
    assert chat.chat(HISTORY) == "ok"
    assert len(chat.calls) == 3


def test_persistent_server_error_raises_instead_of_returning_error_text():
    chat = make_chat(lambda n, r: httpx.Response(503, json={"error": {"message": "down"}}))
    with pytest.raises(LLMRequestFailed) as info:
        chat.chat(HISTORY)
    assert info.value.status_code == 503
    assert len(chat.calls) == 3


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_client_errors_are_not_retried(status):
    chat = make_chat(lambda n, r: httpx.Response(status, json={"error": {"message": "no"}}))
    with pytest.raises(LLMRequestFailed) as info:
        chat.chat(HISTORY)
    assert info.value.status_code == status
    assert len(chat.calls) == 1


def test_dropped_connection_is_retried():
    def handler(n, request):
        if n < 2:
            raise httpx.ConnectError("connection reset", request=request)
        return completion("back")

    chat = make_chat(handler)
    assert chat.chat(HISTORY) == "back"
    assert len(chat.calls) == 2


def test_connection_that_never_recovers_raises():
    def handler(n, request):
        raise httpx.ReadTimeout("slow", request=request)

    chat = make_chat(handler)
    with pytest.raises(LLMRequestFailed):
        chat.chat(HISTORY)
    assert len(chat.calls) == 3


def test_empty_reply_is_retried_then_raises():
    chat = make_chat(lambda n, r: completion(""))
    with pytest.raises(LLMEmptyReply):
        chat.chat(HISTORY)
    assert len(chat.calls) == 3


def test_empty_reply_recovers_on_retry():
    chat = make_chat(lambda n, r: completion("") if n == 1 else completion("text"))
    assert chat.chat(HISTORY) == "text"


@pytest.mark.parametrize("choices", [[], None])
def test_response_without_choices_counts_as_empty(choices):
    body = {"id": "x", "object": "chat.completion", "created": 0, "model": "m", "choices": choices}
    chat = make_chat(lambda n, r: httpx.Response(200, json=body))
    with pytest.raises(LLMEmptyReply):
        chat.chat(HISTORY)
    assert len(chat.calls) == 3


def test_null_content_counts_as_empty():
    chat = make_chat(lambda n, r: completion(None))
    with pytest.raises(LLMEmptyReply):
        chat.chat(HISTORY)


def test_allow_empty_returns_empty_string_without_retry():
    chat = make_chat(lambda n, r: completion(""))
    assert chat.chat(HISTORY, allow_empty=True) == ""
    assert len(chat.calls) == 1


def test_truncated_reply_raises_with_partial_text():
    chat = make_chat(lambda n, r: completion('<json>[{"id"', finish_reason="length"))
    with pytest.raises(LLMTruncated) as info:
        chat.chat(HISTORY, allow_empty=True)
    assert info.value.partial == '<json>[{"id"'
    assert len(chat.calls) == 1


def test_content_filter_raises():
    chat = make_chat(lambda n, r: completion("", finish_reason="content_filter"))
    with pytest.raises(LLMContentFiltered):
        chat.chat(HISTORY, allow_empty=True)
    assert len(chat.calls) == 1


def test_max_tokens_is_sent_only_when_configured():
    chat = make_chat(lambda n, r: completion("ok"))
    chat.chat(HISTORY)
    assert "max_tokens" not in chat.calls[0]

    chat = make_chat(lambda n, r: completion("ok"))
    chat.config.max_tokens = 123
    chat.chat(HISTORY)
    assert chat.calls[0]["max_tokens"] == 123


def test_sdk_exception_types_are_not_leaked():
    chat = make_chat(lambda n, r: httpx.Response(400, json={"error": {"message": "bad"}}))
    with pytest.raises(Exception) as info:
        chat.chat(HISTORY)
    assert not isinstance(info.value, openai.APIError)


def test_validate_completion_rules():
    assert validate_completion("text", "stop") == "text"
    assert validate_completion("", "stop", allow_empty=True) == ""
    assert validate_completion(None, None, allow_empty=True) == ""
    with pytest.raises(LLMEmptyReply):
        validate_completion("  \n", "stop")
    with pytest.raises(LLMTruncated):
        validate_completion("x", "length", allow_empty=True)
    with pytest.raises(LLMContentFiltered):
        validate_completion("x", "content_filter")
