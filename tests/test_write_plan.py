"""Tests for WritePlan and the task import/merge helpers.

The database models, configuration and chat module are replaced with small stand-ins so that the tests need
neither MySQL, Milvus nor a model endpoint.
"""
import importlib
import json
import sys
import types
from typing import List

import pytest
from pydantic import BaseModel, Field

from server.chat.errors import LLMRequestFailed, PlanParseError

TASK_JSON = {"id": "1", "dependent_task_ids": [], "instruction": "Scan 10.0.0.5", "action": "Shell"}
PLAN = f"<json>{json.dumps([TASK_JSON])}</json>"


class FakeTask(BaseModel):
    id: str = Field(None)
    plan_id: str = Field(None)
    sequence: int = Field(...)
    action: str = Field(None)
    instruction: str = Field(None)
    code: List[str] = Field([])
    result: str = Field("")
    is_success: bool = False
    is_finished: bool = False
    dependencies: List[int] = []


@pytest.fixture
def write_plan(monkeypatch):
    chat_calls = []
    replies = []

    def fake_chat(**kwargs):
        chat_calls.append(kwargs)
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    stubs = {
        "config.config": types.SimpleNamespace(Configs=types.SimpleNamespace(
            llm_config=types.SimpleNamespace(max_plan_attempts=3, max_plan_tasks=30),
            kb_config=types.SimpleNamespace(kb_name="kb"))),
        "db.models.plan_model": types.SimpleNamespace(Plan=BaseModel),
        "db.models.task_model": types.SimpleNamespace(TaskModel=object, Task=FakeTask),
        "server.chat.chat": types.SimpleNamespace(_chat=fake_chat),
    }
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delitem(sys.modules, "actions.write_plan", raising=False)
    module = importlib.import_module("actions.write_plan")
    module.chat_calls, module.replies = chat_calls, replies
    yield module
    sys.modules.pop("actions.write_plan", None)


def task_result():
    return types.SimpleNamespace(instruction="Scan", code=["nmap"], result="open: 22")


def test_run_returns_valid_plan_json(write_plan):
    write_plan.replies.append(PLAN)
    out = write_plan.WritePlan(plan_chat_id="p").run("test 10.0.0.5")
    assert json.loads(out) == [TASK_JSON]
    assert write_plan.chat_calls[0]["kb_query"] == "test 10.0.0.5"
    assert "allow_empty" not in write_plan.chat_calls[0]   # an empty initial plan is never acceptable


def test_run_asks_again_after_a_non_plan_reply(write_plan):
    write_plan.replies.extend(["Sorry, I can't.", PLAN])
    out = write_plan.WritePlan(plan_chat_id="p").run("t")
    assert json.loads(out) == [TASK_JSON]
    assert len(write_plan.chat_calls) == 2
    assert "<json></json>" in write_plan.chat_calls[1]["query"]
    assert "<json>[]</json>" not in write_plan.chat_calls[1]["query"]
    assert write_plan.chat_calls[1]["conversation_id"] == "p"


def test_run_raises_when_the_model_never_produces_a_plan(write_plan):
    write_plan.replies.extend(["no", "<json>[]</json>", "<json>{</json>"])
    with pytest.raises(PlanParseError):
        write_plan.WritePlan(plan_chat_id="p").run("t")
    assert len(write_plan.chat_calls) == 3


def test_run_does_not_retry_endpoint_failures(write_plan):
    write_plan.replies.append(LLMRequestFailed("down", status_code=503))
    with pytest.raises(LLMRequestFailed):
        write_plan.WritePlan(plan_chat_id="p").run("t")
    assert len(write_plan.chat_calls) == 1


def test_update_returns_empty_string_when_no_task_remains(write_plan):
    write_plan.replies.append("")
    out = write_plan.WritePlan(plan_chat_id="p").update(task_result(), [], [], "t")
    assert out == ""
    assert write_plan.chat_calls[0]["allow_empty"] is True


def test_update_returns_revised_plan(write_plan):
    write_plan.replies.append(PLAN)
    out = write_plan.WritePlan(plan_chat_id="p").update(task_result(), ["a"], [], "t")
    assert json.loads(out) == [TASK_JSON]


def test_update_repair_prompt_allows_an_empty_list(write_plan):
    write_plan.replies.extend(["maybe nothing", PLAN])
    write_plan.WritePlan(plan_chat_id="p").update(task_result(), [], [], "t")
    assert "<json>[]</json>" in write_plan.chat_calls[1]["query"]
    assert write_plan.chat_calls[1]["allow_empty"] is True


def test_import_and_merge_use_validated_tasks(write_plan):
    plan_json = json.dumps([
        {"id": "1", "dependent_task_ids": [], "instruction": "scan", "action": "Shell"},
        {"id": "2", "dependent_task_ids": ["1", "2"], "instruction": "exploit", "action": "Shell"},
    ])
    tasks = write_plan.import_tasks_from_json("plan-1", write_plan.parse_plan_json(plan_json))
    assert [t.instruction for t in tasks] == ["scan", "exploit"]
    assert tasks[1].dependencies == [0]            # the task's reference to itself is dropped

    tasks[0].is_finished = tasks[0].is_success = True
    merged = write_plan.merge_tasks_from_json("plan-1", write_plan.parse_plan_json(plan_json), tasks)
    assert [t.instruction for t in merged] == ["scan", "exploit"]
    assert merged[0] is tasks[0]


def test_merge_keeps_completed_tasks_missing_from_the_new_plan(write_plan):
    old = [FakeTask(plan_id="p", sequence=0, instruction="scan", action="Shell", is_finished=True, is_success=True)]
    new = write_plan.parse_plan_json(json.dumps([{"id": "1", "instruction": "exploit", "action": "Shell"}]))
    merged = write_plan.merge_tasks_from_json("p", new, old)
    assert [t.instruction for t in merged] == ["scan", "exploit"]
