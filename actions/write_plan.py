from typing import List, Dict

from pydantic import BaseModel

from actions.plan_schema import get_valid_plan, parse_plan_json
from config.config import Configs
from prompts.prompt import DeepPentestPrompt
from db.models.plan_model import Plan
from db.models.task_model import TaskModel, Task
from server.chat.chat import _chat


class WritePlan(BaseModel):
    plan_chat_id: str

    def _plan_limits(self) -> dict:
        llm = Configs.llm_config
        return dict(attempts=llm.max_plan_attempts, max_tasks=llm.max_plan_tasks)

    def run(self, init_description) -> str:
        """Ask for the initial plan; return its JSON text. Raises PlanParseError if no usable plan arrives."""
        def ask(feedback):
            if feedback is None:
                return _chat(query=DeepPentestPrompt.write_plan, conversation_id=self.plan_chat_id,
                             kb_name=Configs.kb_config.kb_name, kb_query=init_description)
            return _chat(query=feedback, conversation_id=self.plan_chat_id)

        return get_valid_plan(ask, DeepPentestPrompt.repair_plan, **self._plan_limits())

    def update(self, task_result, success_task, fail_task, init_description) -> str:
        """Ask for the revised plan; return its JSON text.

        Returns "" when the model reports that no task remains for this stage (an empty reply or list),
        which is the answer the update prompt asks for in that case.
        """
        def ask(feedback):
            if feedback is None:
                return _chat(
                    query=DeepPentestPrompt.update_plan.format(current_task=task_result.instruction,
                                                              init_description=init_description,
                                                              current_code=task_result.code,
                                                              task_result=task_result.result,
                                                              success_task=success_task,
                                                              fail_task=fail_task),
                    conversation_id=self.plan_chat_id,
                    kb_name=Configs.kb_config.kb_name,
                    kb_query=task_result.instruction,
                    allow_empty=True,
                )
            return _chat(query=feedback, conversation_id=self.plan_chat_id, allow_empty=True)

        return get_valid_plan(ask, DeepPentestPrompt.repair_plan_update, allow_empty=True, **self._plan_limits())


def parse_tasks(response: str, current_plan: Plan):
    tasks = import_tasks_from_json(current_plan.id, parse_plan_json(response))

    current_plan.tasks = tasks

    return current_plan


def merge_tasks(response: str, current_plan: Plan):

    tasks = merge_tasks_from_json(current_plan.id, parse_plan_json(response), current_plan.tasks)

    current_plan.tasks = tasks

    return current_plan


def import_tasks_from_json(plan_id: str, tasks_json: List[Dict]) -> List[TaskModel]:
    tasks = []
    for idx, task_data in enumerate(tasks_json):
        task = Task(
            plan_id=plan_id,
            sequence=idx,
            action=task_data['action'],
            instruction=task_data['instruction'],
            dependencies=[i for i, t in enumerate(tasks_json)
                          if i != idx and t['id'] in task_data['dependent_task_ids']]
        )

        tasks.append(task)
    return tasks


def merge_tasks_from_json(plan_id: str, new_tasks_json: List[Dict], old_tasks: List[Task]) -> List[Task]:
    # 获取所有已完成且成功的任务
    completed_tasks_map = {
        task.instruction: task
        for task in old_tasks
        if task.is_finished and task.is_success
    }

    merged_tasks = []

    for instruction, completed_task in completed_tasks_map.items():
        found = False
        for task_data in new_tasks_json:
            if task_data['instruction'] == instruction:
                found = True
                break
        if not found:
            completed_task.sequence = len(merged_tasks)
            completed_task.dependencies = []
            merged_tasks.append(completed_task)

    new_task_id_to_idx = {
        task_data.get('id'): idx+len(merged_tasks)
        for idx, task_data in enumerate(new_tasks_json)
    }
    for idx, task_data in enumerate(new_tasks_json):
        instruction = task_data['instruction']
        sequence = len(merged_tasks)

        if instruction in completed_tasks_map:
            existing_task = completed_tasks_map[instruction]
            existing_task.sequence = sequence
            existing_task.dependencies = [
                new_task_id_to_idx[dep_id]
                for dep_id in task_data['dependent_task_ids']
                if dep_id in new_task_id_to_idx
            ]
            merged_tasks.append(existing_task)
        else:
            new_task = Task(
                plan_id=plan_id,
                sequence=sequence,
                action=task_data['action'],
                instruction=task_data['instruction'],
                dependencies=[
                    new_task_id_to_idx[dep_id]
                    for dep_id in task_data['dependent_task_ids']
                    if dep_id in new_task_id_to_idx
                ],
            )
            merged_tasks.append(new_task)

    return merged_tasks