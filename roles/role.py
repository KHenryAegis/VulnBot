import traceback
from typing import Any, ClassVar
from pydantic import Field, BaseModel
from actions.plan_summary import PlannerSummary
from actions.planner import Planner
from actions.write_code import WriteCode
from db.models.plan_model import Plan
from db.repository.plan_repository import get_planner_by_id, add_plan_to_db
from db.repository.task_repository import add_task_to_plan
from prompts.prompt import DeepPentestPrompt
from server.chat.chat import _chat
from server.chat.errors import EmptyPlanError, LLMError, PlanParseError
from utils.log_common import build_logger

logger = build_logger()

class Role(BaseModel):
    name: str
    goal: str
    tools: str
    prompt: ClassVar
    max_interactions: int = 5
    previous_summary: PlannerSummary = Field(default_factory=PlannerSummary)
    planner: Planner = Field(default_factory=Planner)
    chat_counter: int = 0
    plan_chat_id: str = ""
    react_chat_id: str = ""
    console: Any = None

    def get_summary(self, history_planner_ids):
        self.previous_summary = PlannerSummary(history_planner_ids=history_planner_ids)
        return self.previous_summary.get_summary()

    def put_message(self, message):
        add_task_to_plan(self.planner.current_plan.tasks)
        # To be implemented in each subclass
        pass

    def _react(self, next_task):
        try:
            self.chat_counter += 1
            writer = WriteCode(next_task=next_task, action=self.planner.current_plan.current_task.action)
            result = writer.run()
            self.console.print("---------- Execute Result ---------", style="bold green")
            logger.info(result.response)
            self.console.print("---------- Execute Result End ---------", style="bold green")
            self.planner.current_plan.current_task.code = result.context["code"]
            if len(result.response) >= 8192:
                response, _ = _chat(query=DeepPentestPrompt.summary_result + str(result.response), summary=False)

                logger.info(f"result summary: {response}")
                result.response = response

            return self.planner.update_plan(result.response)
        except (LLMError, PlanParseError):
            # The endpoint failed or the plan is unusable: stop the run instead of ending it quietly.
            raise
        except Exception as e:
            print(e)
            print(traceback.format_exc())

    def _plan(self, session):
        if session.current_planner_id != '':
            self.planner = Planner(current_plan=get_planner_by_id(session.current_planner_id), init_description=session.init_description)
        else:
            with self.console.status("[bold green] Initializing DeepPentest Sessions...") as status:
                try:
                    context = self.get_summary(session.history_planner_ids)
                    (text_0, self.plan_chat_id) = _chat(
                        query=self.prompt.init_plan_prompt.format(init_description=session.init_description,
                                                                  goal=self.goal,
                                                                  tools=self.tools,
                                                                  context=context)
                    )
                    (text_1, self.react_chat_id) = _chat(query=self.prompt.init_reasoning_prompt)
                except Exception as e:
                    self.console.print(f"Failed to initialize chat sessions: {e}", style="bold red")
                    raise
            plan = Plan(goal=self.goal, plan_chat_id=self.plan_chat_id, react_chat_id=self.react_chat_id, current_task_sequence=0)
            plan = add_plan_to_db(plan)
            self.console.print("Plan Initialized.", style="bold green")
            session.current_planner_id = plan.id
            self.planner = Planner(current_plan=plan, init_description=session.init_description)


        return self.planner.plan()

    def run(self, session):
        next_task = self._plan(session)
        if next_task is None:
            raise EmptyPlanError(f"The {self.name} plan has no task to execute.")
        while self.chat_counter < self.max_interactions:
            next_task = self._react(next_task)
            if next_task is None:
                break
        self.put_message(session)