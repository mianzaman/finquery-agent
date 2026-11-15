"""Stand-ins for chat models, so no test calls a real LLM."""

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field


def reply(text: str, input_tokens: int = 100, output_tokens: int = 20) -> AIMessage:
    """A model reply with token counts, as a provider would return it."""
    usage = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    return AIMessage(content=text, usage_metadata=usage)


class ScriptedChat(BaseChatModel):
    """Returns the given replies in order (or raises them) and records what it was sent."""

    replies: list[Any]
    model_name: str = "test/model:free"
    seen: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> ChatResult:
        self.seen.append(list(messages))
        if not self.replies:
            raise AssertionError("the model was called more often than the test scripted")
        item = self.replies.pop(0)
        if isinstance(item, Exception):
            raise item
        return ChatResult(generations=[ChatGeneration(message=item)])
