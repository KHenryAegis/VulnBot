import json

import pytest

from actions.plan_schema import (MAX_ACTION_LEN, MAX_INSTRUCTION_LEN, extract_json_block, get_valid_plan,
                                 parse_plan_json)
from server.chat.errors import (EmptyPlanError, LLMContentFiltered, LLMEmptyReply, LLMTruncated,
                                PlanParseError)

TASK = {"id": "1", "dependent_task_ids": [], "instruction": "Scan 10.0.0.5 with nmap", "action": "Shell"}
PLAN = f"<json>{json.dumps([TASK])}</json>"
REPAIR = "Unusable: {reason}"


# --- extract_json_block -----------------------------------------------------------------------

@pytest.mark.parametrize("reply", [None, "", "   \n"])
def test_extract_rejects_empty_reply(reply):
    with pytest.raises(PlanParseError, match="empty"):
        extract_json_block(reply)


def test_extract_rejects_reply_without_json_block():
    with pytest.raises(PlanParseError, match="no <json>"):
        extract_json_block("I cannot help with that.")


def test_extract_reads_json_tag_and_fenced_block():
    assert json.loads(extract_json_block(f"Plan:\n{PLAN}\nDone"))[0]["id"] == "1"
    assert json.loads(extract_json_block("```json\n" + json.dumps([TASK]) + "\n```"))[0]["id"] == "1"


# --- parse_plan_json --------------------------------------------------------------------------

def test_parse_valid_plan():
    tasks = parse_plan_json(json.dumps([TASK]))
    assert tasks == [{"id": "1", "action": "Shell", "instruction": TASK["instruction"], "dependent_task_ids": []}]


def test_parse_malformed_json_names_the_position():
    with pytest.raises(PlanParseError, match="not valid JSON"):
        parse_plan_json('[{"id": "1", "action": "Shell"')


def test_parse_issue_example_expecting_colon_delimiter():
    # JSONDecodeError "Expecting ':' delimiter" from the report becomes a PlanParseError.
    with pytest.raises(PlanParseError, match="Expecting ':' delimiter"):
        parse_plan_json('[{"id" "1"}]')


def test_parse_repairs_invalid_escapes_in_instructions():
    text = '[{"id": "1", "action": "Shell", "instruction": "echo \\@ \\! done", "dependent_task_ids": []}]'
    assert parse_plan_json(text)[0]["instruction"] == "echo \\@ \\! done"


@pytest.mark.parametrize("missing", ["action", "instruction"])
def test_parse_rejects_missing_keys(missing):
    task = {k: v for k, v in TASK.items() if k != missing}
    with pytest.raises(PlanParseError, match=f'no "{missing}"'):
        parse_plan_json(json.dumps([task]))


def test_parse_defaults_missing_id_and_dependencies():
    task = {"action": "Shell", "instruction": "do it"}
    assert parse_plan_json(json.dumps([task, task])) == [
        {"id": "1", "action": "Shell", "instruction": "do it", "dependent_task_ids": []},
        {"id": "2", "action": "Shell", "instruction": "do it", "dependent_task_ids": []},
    ]


def test_parse_rejects_non_object_task_and_non_list_plan():
    with pytest.raises(PlanParseError, match="not a JSON object"):
        parse_plan_json('["nmap"]')
    with pytest.raises(PlanParseError, match="must be a JSON list"):
        parse_plan_json('"nmap"')
    with pytest.raises(PlanParseError, match="not a list"):
        parse_plan_json(json.dumps([{**TASK, "dependent_task_ids": "1"}]))


@pytest.mark.parametrize("text", ["", "  ", "[]"])
def test_parse_empty_plan_is_an_error_unless_allowed(text):
    with pytest.raises(EmptyPlanError):
        parse_plan_json(text)
    assert parse_plan_json(text, allow_empty=True) == []


def test_parse_bounds_oversized_plans():
    many = json.dumps([TASK] * 31)
    with pytest.raises(PlanParseError, match="31 tasks"):
        parse_plan_json(many)
    huge = json.dumps([{**TASK, "id": str(i)} for i in range(45000)])
    with pytest.raises(PlanParseError, match="characters long"):
        parse_plan_json(huge)


def test_parse_column_limits():
    long_instruction = {**TASK, "instruction": "x" * (MAX_INSTRUCTION_LEN + 100)}
    assert len(parse_plan_json(json.dumps([long_instruction]))[0]["instruction"]) == MAX_INSTRUCTION_LEN
    with pytest.raises(PlanParseError, match="action"):
        parse_plan_json(json.dumps([{**TASK, "action": "a" * (MAX_ACTION_LEN + 1)}]))


# --- get_valid_plan ---------------------------------------------------------------------------

def scripted(*replies):
    """An ask() that returns or raises each scripted item in turn and records the feedback it was given."""
    feedback_seen, items = [], list(replies)

    def ask(feedback):
        feedback_seen.append(feedback)
        item = items.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    ask.feedback = feedback_seen
    return ask


def test_valid_plan_needs_one_call():
    ask = scripted(PLAN)
    assert json.loads(get_valid_plan(ask, REPAIR, attempts=3))[0]["id"] == "1"
    assert ask.feedback == [None]


@pytest.mark.parametrize("bad", [
    "I cannot help with that.",             # non-plan reply
    "<json>[{oops</json>",                   # malformed JSON
    '<json>[{"id": "1"}]</json>',            # missing keys
    "<json>[]</json>",                       # empty task list
    "",                                      # empty reply
    LLMEmptyReply("empty"), LLMTruncated("cut off"), LLMContentFiltered("blocked"),
])
def test_bad_reply_is_repaired_on_the_next_attempt(bad):
    ask = scripted(bad, PLAN)
    assert json.loads(get_valid_plan(ask, REPAIR, attempts=3))[0]["id"] == "1"
    assert ask.feedback[0] is None
    assert ask.feedback[1].startswith("Unusable: ")


def test_gives_up_after_the_configured_attempts():
    ask = scripted("nope", "nope", "nope", PLAN)
    with pytest.raises(PlanParseError, match="after 3 attempts"):
        get_valid_plan(ask, REPAIR, attempts=3)
    assert len(ask.feedback) == 3


def test_empty_plan_error_type_is_kept_when_giving_up():
    ask = scripted("<json>[]</json>", "<json>[]</json>")
    with pytest.raises(EmptyPlanError):
        get_valid_plan(ask, REPAIR, attempts=2)


def test_endpoint_failures_are_not_swallowed():
    from server.chat.errors import LLMRequestFailed
    ask = scripted(LLMRequestFailed("down", status_code=503))
    with pytest.raises(LLMRequestFailed):
        get_valid_plan(ask, REPAIR, attempts=3)
    assert len(ask.feedback) == 1


@pytest.mark.parametrize("reply", ["", None, "<json></json>", "<json>[]</json>"])
def test_update_treats_an_empty_answer_as_no_further_tasks(reply):
    ask = scripted(reply)
    assert get_valid_plan(ask, REPAIR, attempts=3, allow_empty=True) == ""
    assert len(ask.feedback) == 1


def test_update_still_validates_a_non_empty_answer():
    ask = scripted("no tasks apply, I think", PLAN)
    assert json.loads(get_valid_plan(ask, REPAIR, attempts=3, allow_empty=True))[0]["id"] == "1"
