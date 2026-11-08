import time
from abc import ABC
from typing import Callable, List, Optional

import httpx
import openai
from ollama import Client
from openai import OpenAI
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential

from server.chat.errors import (LLMBadReply, LLMEmptyReply, LLMRequestFailed,
                                validate_completion)

RETRYABLE_STATUS = {408, 409, 429}


def is_transient(exc: BaseException) -> bool:
    """True for failures that a retry can plausibly fix."""
    if isinstance(exc, LLMEmptyReply):
        return True
    if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError, httpx.TransportError)):
        return True
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code in RETRYABLE_STATUS or exc.status_code >= 500
    return False


class OpenAIChat(ABC):
    # Replaced in tests so that retries do not sleep.
    sleep: Callable[[float], None] = staticmethod(time.sleep)

    def __init__(self, config, client: Optional[OpenAI] = None):
        self.config = config
        # The SDK's own retries are disabled so that the policy below is the only one.
        self.client = client or OpenAI(api_key=self.config.api_key, base_url=self.config.base_url,
                                       timeout=config.timeout, max_retries=0)
        self.model_name = self.config.llm_model_name

    def _complete(self, history: List, allow_empty: bool) -> str:
        kwargs = {}
        if getattr(self.config, "max_tokens", 0) > 0:
            kwargs["max_tokens"] = self.config.max_tokens
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=history,
            temperature=self.config.temperature,
            **kwargs,
        )
        if not response.choices:
            raise LLMEmptyReply("The endpoint returned no choices")
        choice = response.choices[0]
        return validate_completion(choice.message.content, choice.finish_reason, allow_empty)

    def chat(self, history: List, allow_empty: bool = False) -> str:
        """Return the model's reply.

        Transient failures (dropped connections, timeouts, 408/409/429/5xx,
        empty replies) are retried with exponential backoff. Anything else is
        raised as an LLMError instead of being returned as reply text.
        """
        retrying = Retrying(
            stop=stop_after_attempt(max(1, getattr(self.config, "max_retries", 3))),
            wait=wait_exponential(multiplier=1, min=1, max=30),
            retry=retry_if_exception(is_transient),
            sleep=lambda seconds: type(self).sleep(seconds),
            reraise=True,
        )
        try:
            return retrying(self._complete, history, allow_empty)
        except LLMBadReply:
            raise
        except openai.APIStatusError as e:
            raise LLMRequestFailed(f"Endpoint returned HTTP {e.status_code}: {e.message}",
                                   status_code=e.status_code) from e
        except (openai.APIConnectionError, httpx.TransportError) as e:
            raise LLMRequestFailed(f"Could not reach the endpoint: {e}") from e


class OllamaChat(ABC):
    sleep: Callable[[float], None] = staticmethod(time.sleep)

    def __init__(self, config, client: Optional[Client] = None):
        self.config = config
        self.client = client or Client(host=self.config.base_url)
        self.model_name = self.config.llm_model_name

    def _complete(self, history: List[dict], allow_empty: bool) -> str:
        options = {"temperature": self.config.temperature}
        if getattr(self.config, "max_tokens", 0) > 0:
            options["num_predict"] = self.config.max_tokens
        response = self.client.chat(
            model=self.model_name,
            messages=history,
            options=options,
            keep_alive=-1
        )
        return validate_completion(response["message"]["content"], response.get("done_reason"), allow_empty)

    def chat(self, history: List[dict], allow_empty: bool = False) -> str:
        retrying = Retrying(
            stop=stop_after_attempt(max(1, getattr(self.config, "max_retries", 3))),
            wait=wait_exponential(multiplier=1, min=1, max=30),
            retry=retry_if_exception(lambda e: isinstance(e, (LLMEmptyReply, httpx.TransportError, ConnectionError))),
            sleep=lambda seconds: type(self).sleep(seconds),
            reraise=True,
        )
        try:
            return retrying(self._complete, history, allow_empty)
        except LLMBadReply:
            raise
        except Exception as e:
            raise LLMRequestFailed(f"Ollama request failed: {e}") from e
