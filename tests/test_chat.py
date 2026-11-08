"""Tests for _chat with the database, knowledge base and configuration replaced by stand-ins."""
import importlib
import logging
import sys
import types
from enum import StrEnum

import pytest

from server.chat.errors import LLMEmptyReply, LLMError, LLMRequestFailed


class LLMType(StrEnum):
    OPENAI = "openai"
    OLLAMA = "ollama"


@pytest.fixture
def chat_module(monkeypatch):
    state = types.SimpleNamespace(
        docs=[], reranked=None, rerank_calls=0, sent=[], saved=[], reply="model reply", error=None,
        llm_model="openai", enable_rag=True)

    class FakeClient:
        def __init__(self, config):
            pass

        def chat(self, history, allow_empty=False):
            state.sent.append((history, allow_empty))
            if state.error:
                raise state.error
            return state.reply

    class FakeReranker:
        def __init__(self, **kwargs):
            pass

        def compress_documents(self, documents, query):
            state.rerank_calls += 1
            return state.reranked if state.reranked is not None else documents

    async def run_in_threadpool(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    configs = types.SimpleNamespace(
        basic_config=types.SimpleNamespace(enable_rag=True),
        kb_config=types.SimpleNamespace(top_k=3, score_threshold=0.5, top_n=2),
        llm_config=types.SimpleNamespace(llm_model="openai", llm_model_name="m", context_length=1000,
                                         history_len=5, rerank_model="r"))
    state.configs = configs

    stubs = {
        "starlette": types.ModuleType("starlette"),
        "starlette.concurrency": types.SimpleNamespace(run_in_threadpool=run_in_threadpool),
        "config.config": types.SimpleNamespace(Configs=configs),
        "db.repository.conversation_repository": types.SimpleNamespace(
            add_conversation_to_db=lambda model, conversation_id: conversation_id or "new-conversation"),
        "db.repository.message_repository": types.SimpleNamespace(
            get_conversation_messages=lambda conversation_id: [],
            add_message_to_db=lambda *args: state.saved.append(args)),
        "rag.kb.api.kb_doc_api": types.SimpleNamespace(search_docs=lambda **kwargs: state.docs),
        "rag.reranker.reranker": types.SimpleNamespace(LangchainReranker=FakeReranker),
        "server.utils.utils": types.SimpleNamespace(LLMType=LLMType, replace_ip_with_targetip=lambda text: text),
        "utils.log_common": types.SimpleNamespace(build_logger=lambda: logging.getLogger("vulnbot-test")),
        "server.chat.llm_clients": types.SimpleNamespace(OpenAIChat=FakeClient, OllamaChat=FakeClient),
    }
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delitem(sys.modules, "server.chat.chat", raising=False)
    module = importlib.import_module("server.chat.chat")
    yield module, state
    sys.modules.pop("server.chat.chat", None)


def test_returns_reply_and_conversation_id(chat_module):
    chat, state = chat_module
    assert chat._chat("hello") == ("model reply", "new-conversation")
    assert chat._chat("hello", conversation_id="c1") == "model reply"
    assert len(state.saved) == 2


def test_endpoint_failure_propagates_and_nothing_is_saved(chat_module):
    chat, state = chat_module
    state.error = LLMRequestFailed("down", status_code=503)
    with pytest.raises(LLMRequestFailed):
        chat._chat("hello", conversation_id="c1")
    assert state.saved == []


def test_bad_reply_error_propagates(chat_module):
    chat, state = chat_module
    state.error = LLMEmptyReply("empty")
    with pytest.raises(LLMEmptyReply):
        chat._chat("hello", conversation_id="c1")


def test_allow_empty_reaches_the_client(chat_module):
    chat, state = chat_module
    state.reply = ""
    assert chat._chat("hello", conversation_id="c1", allow_empty=True) == ""
    assert state.sent[0][1] is True
    chat._chat("hello", conversation_id="c1")
    assert state.sent[1][1] is False


def test_unsupported_model_type_raises(chat_module):
    chat, state = chat_module
    state.configs.llm_config.llm_model = "mystery"
    with pytest.raises(LLMError, match="Unsupported model type"):
        chat._chat("hello")


def test_empty_retrieval_warns_and_continues_without_context(chat_module, caplog):
    chat, state = chat_module
    state.docs = []
    with caplog.at_level(logging.WARNING, logger="vulnbot-test"):
        reply = chat._chat("do the task", kb_name="kb", kb_query="nmap scan", conversation_id="c1")
    assert reply == "model reply"
    assert state.rerank_calls == 0                       # nothing to rerank
    assert state.sent[0][0][-1]["content"] == "do the task"   # the prompt is sent unchanged
    assert "No knowledge-base documents matched" in caplog.text


def test_retrieval_emptied_by_reranking_warns(chat_module, caplog):
    chat, state = chat_module
    state.docs, state.reranked = [{"page_content": "x"}], []
    with caplog.at_level(logging.WARNING, logger="vulnbot-test"):
        chat._chat("do the task", kb_name="kb", kb_query="q", conversation_id="c1")
    assert state.rerank_calls == 1
    assert "No knowledge-base documents matched" in caplog.text


def test_retrieved_context_is_added_to_the_prompt(chat_module):
    chat, state = chat_module
    state.docs = [{"page_content": "use nmap -sV"}]
    chat._chat("do the task", kb_name="kb", kb_query="q", conversation_id="c1")
    assert "use nmap -sV" in state.sent[0][0][-1]["content"]
