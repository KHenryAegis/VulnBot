"""Tests for how Role reacts to LLM and plan failures."""
import importlib
import logging
import sys
import types
from typing import List

import pytest
from pydantic import BaseModel, Field

from server.chat.errors import EmptyPlanError, LLMRequestFailed, PlanParseError


class FakeSummary(BaseModel):
    history_planner_ids: List[str] = Field(default_factory=list)

    def get_summary(self):
        return ""


class FakePlanner(BaseModel):
    current_plan: object = None


@pytest.fixture
def role_module(monkeypatch):
    state = types.SimpleNamespace(write_code=None, chat=None, plan_result="first task", update_result=None)

    class FakeWriteCode:
        def __init__(self, next_task, action):
            pass

        def run(self):
            return state.write_code()

    stubs = {
        "actions.plan_summary": types.SimpleNamespace(PlannerSummary=FakeSummary),
        "actions.planner": types.SimpleNamespace(Planner=FakePlanner),
        "actions.write_code": types.SimpleNamespace(WriteCode=FakeWriteCode),
        "db.models.plan_model": types.SimpleNamespace(Plan=lambda **kw: types.SimpleNamespace(id="p", **kw)),
        "db.repository.plan_repository": types.SimpleNamespace(get_planner_by_id=lambda i: None,
                                                               add_plan_to_db=lambda plan: plan),
        "db.repository.task_repository": types.SimpleNamespace(add_task_to_plan=lambda tasks: None),
        "server.chat.chat": types.SimpleNamespace(_chat=lambda **kw: state.chat(**kw)),
        "utils.log_common": types.SimpleNamespace(build_logger=lambda: logging.getLogger("role-test")),
    }
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delitem(sys.modules, "roles.role", raising=False)
    module = importlib.import_module("roles.role")
    yield module, state
    sys.modules.pop("roles.role", None)


class Console:
    def print(self, *args, **kwargs):
        pass

    def status(self, *args, **kwargs):
        import contextlib
        return contextlib.nullcontext()


def make_role(module, planner=None):
    class TestRole(module.Role):
        prompt = types.SimpleNamespace(init_plan_prompt="{init_description}{goal}{tools}{context}",
                                       init_reasoning_prompt="reason")
    role = TestRole(name="Test", goal="g", tools="t", console=Console())
    if planner is not None:
        object.__setattr__(role, "planner", planner)
    return role


def current_task_planner():
    task = types.SimpleNamespace(action="Shell", code=None)
    plan = types.SimpleNamespace(current_task=task)
    return types.SimpleNamespace(current_plan=plan)


def test_run_stops_when_the_plan_has_no_task(role_module):
    module, state = role_module
    role = make_role(module)
    role._plan = lambda session: None
    with pytest.raises(EmptyPlanError):
        role.run(types.SimpleNamespace())


def test_react_propagates_endpoint_failure(role_module):
    module, state = role_module
    role = make_role(module, current_task_planner())

    def failing():
        raise LLMRequestFailed("down", status_code=503)

    state.write_code = failing
    with pytest.raises(LLMRequestFailed):
        role._react("task")


def test_react_propagates_plan_failure(role_module):
    module, state = role_module
    role = make_role(module, current_task_planner())

    def failing():
        raise PlanParseError("no plan")

    state.write_code = failing
    with pytest.raises(PlanParseError):
        role._react("task")


def test_react_still_ends_the_loop_on_other_errors(role_module):
    module, state = role_module
    role = make_role(module, current_task_planner())

    def failing():
        raise RuntimeError("shell closed")

    state.write_code = failing
    assert role._react("task") is None


def test_plan_propagates_failure_to_start_the_chat_sessions(role_module):
    module, state = role_module

    def failing(**kwargs):
        raise LLMRequestFailed("down", status_code=503)

    state.chat = failing
    role = make_role(module)
    session = types.SimpleNamespace(current_planner_id="", init_description="t", history_planner_ids=[])
    with pytest.raises(LLMRequestFailed):
        role._plan(session)
