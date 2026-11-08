"""Exceptions raised when the LLM endpoint or a model reply cannot be used.

Callers used to receive the text "**ERROR**: ..." in place of a model reply and
pass it on as if the model had written it. These exceptions make the failure
explicit so each layer can retry, repair, or stop the run.
"""
from typing import Optional


class LLMError(Exception):
    """Base class for LLM endpoint failures and unusable replies."""


class LLMRequestFailed(LLMError):
    """The endpoint returned an error or could not be reached, after retries."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class LLMBadReply(LLMError):
    """The endpoint answered, but the reply is not usable."""


class LLMEmptyReply(LLMBadReply):
    """The reply contained no text."""


class LLMTruncated(LLMBadReply):
    """Generation stopped at the token limit (finish_reason == "length")."""

    def __init__(self, message: str, partial: str = ""):
        super().__init__(message)
        self.partial = partial


class LLMContentFiltered(LLMBadReply):
    """The provider blocked the reply (finish_reason == "content_filter")."""


class PlanParseError(ValueError):
    """The model's plan could not be turned into a valid task list."""


class EmptyPlanError(PlanParseError):
    """A plan with no executable task where at least one is required."""


def validate_completion(content: Optional[str], finish_reason: Optional[str],
                        allow_empty: bool = False) -> str:
    """Return the reply text, or raise the LLMBadReply that describes the problem.

    An empty reply is accepted only when ``allow_empty`` is set and the model
    stopped normally. Some prompts define an empty reply as the answer.
    """
    if finish_reason == "content_filter":
        raise LLMContentFiltered("The reply was blocked by the provider's content filter")
    if finish_reason == "length":
        raise LLMTruncated("The reply was cut off at the token limit", partial=content or "")
    if content is None or not content.strip():
        if allow_empty and finish_reason in (None, "stop"):
            return ""
        raise LLMEmptyReply("The model returned an empty reply")
    return content
