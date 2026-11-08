"""Extraction and validation of the task list that the model writes as its plan.

Everything here is independent of the database and the model client, so a bad
reply is rejected, with a reason the model can act on, before it reaches
``json.loads`` or the task tables.
"""
import json
import logging
import re
from typing import Callable, Dict, List, Optional

from server.chat.errors import EmptyPlanError, LLMBadReply, PlanParseError

logger = logging.getLogger(__name__)

# Column sizes of the tasks table (db/models/task_model.py).
MAX_ACTION_LEN = 255
MAX_INSTRUCTION_LEN = 512
# The plan prompt asks for 1 to 5 tasks; this only bounds runaway output.
MAX_PLAN_TASKS = 30
MAX_PLAN_CHARS = 100_000

_JSON_TAG = re.compile(r"<json>(.*?)</json>", re.DOTALL)
_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_BAD_ESCAPE = re.compile(r"\\([@!])")


def extract_json_block(reply: Optional[str]) -> str:
    """Return the text inside the reply's <json></json> block (or a fenced block).

    Raises PlanParseError when the reply has no such block.
    """
    if reply is None or not reply.strip():
        raise PlanParseError("The reply is empty.")
    match = _JSON_TAG.search(reply) or _JSON_FENCE.search(reply)
    if match is None:
        raise PlanParseError("The reply has no <json></json> block.")
    return match.group(1)


def parse_plan_json(text: str, *, allow_empty: bool = False, max_tasks: int = MAX_PLAN_TASKS,
                    max_chars: int = MAX_PLAN_CHARS) -> List[Dict]:
    """Parse and validate a task list; return tasks with ``id``, ``action``, ``instruction``,
    ``dependent_task_ids`` set.

    An empty list raises EmptyPlanError unless ``allow_empty`` is set. Instructions longer than
    the database column are truncated; overlong actions, missing fields and oversized plans are
    rejected with a PlanParseError that names the problem.
    """
    if text is None:
        raise PlanParseError("The plan is empty.")
    if len(text) > max_chars:
        raise PlanParseError(f"The plan is {len(text)} characters long; the limit is {max_chars}.")
    if not text.strip():
        if allow_empty:
            return []
        raise EmptyPlanError("The plan is empty.")

    # Invalid escape sequences such as "\@" or "\!" are common in model output.
    cleaned = _BAD_ESCAPE.sub(r"\\\\\1", text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise PlanParseError(f"The plan is not valid JSON ({e.msg} at line {e.lineno}, column {e.colno}).") from e

    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise PlanParseError("The plan must be a JSON list of task objects.")
    if not data:
        if allow_empty:
            return []
        raise EmptyPlanError("The plan has no tasks.")
    if len(data) > max_tasks:
        raise PlanParseError(f"The plan has {len(data)} tasks; the limit is {max_tasks}.")

    tasks: List[Dict] = []
    for idx, item in enumerate(data):
        where = f"Task {idx + 1}"
        if not isinstance(item, dict):
            raise PlanParseError(f"{where} is not a JSON object.")
        for key in ("action", "instruction"):
            value = item.get(key)
            if not isinstance(value, str) or not value.strip():
                raise PlanParseError(f'{where} has no "{key}" text.')
        action = item["action"].strip()
        if len(action) > MAX_ACTION_LEN:
            raise PlanParseError(f'{where} has an "action" of {len(action)} characters; the limit is {MAX_ACTION_LEN}.')
        instruction = item["instruction"].strip()
        if len(instruction) > MAX_INSTRUCTION_LEN:
            logger.warning(f"{where}: instruction truncated from {len(instruction)} to {MAX_INSTRUCTION_LEN} characters")
            instruction = instruction[:MAX_INSTRUCTION_LEN]
        deps = item.get("dependent_task_ids", [])
        if not isinstance(deps, list):
            raise PlanParseError(f'{where} has a "dependent_task_ids" that is not a list.')
        tasks.append({
            "id": str(item.get("id", idx + 1)),
            "action": action,
            "instruction": instruction,
            "dependent_task_ids": [str(d) for d in deps],
        })
    return tasks


def get_valid_plan(ask: Callable[[Optional[str]], str], repair_prompt: str, *, attempts: int,
                   allow_empty: bool = False, **limits) -> str:
    """Ask the model for a plan until one validates; return its JSON text.

    ``ask(feedback)`` sends a request and returns the reply text; ``feedback`` is None on the first call
    and ``repair_prompt`` filled with the reason after a rejected reply. Unusable replies (empty,
    truncated, filtered, no JSON, invalid plan) are retried up to ``attempts`` times in total, then the
    last PlanParseError is raised. Endpoint failures are not retried here.

    With ``allow_empty`` an empty reply or an empty list means "no further tasks" and returns "".
    """
    attempts = max(1, attempts)
    feedback: Optional[str] = None
    last_error: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            reply = ask(feedback)
            if allow_empty and (reply is None or not reply.strip()):
                return ""
            block = extract_json_block(reply)
            tasks = parse_plan_json(block, allow_empty=allow_empty, **limits)
            return block if tasks else ""
        except (PlanParseError, LLMBadReply) as e:
            last_error = e
            logger.warning(f"Plan attempt {attempt}/{attempts} rejected: {e}")
            feedback = repair_prompt.format(reason=e)
    error_type = EmptyPlanError if isinstance(last_error, EmptyPlanError) else PlanParseError
    raise error_type(f"No usable plan after {attempts} attempts: {last_error}") from last_error
