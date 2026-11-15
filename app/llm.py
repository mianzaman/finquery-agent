"""The only place that creates chat or embedding models and picks the provider.

Everything else asks this module for a model, or calls `structured()`. Switching between the
free OpenRouter models and paid OpenAI or Gemini is a change in `.env` only.

Request budget (free tier: 20 requests a minute, 50 a day):
- every model shares one rate limiter set from `LLM_MAX_RPM`;
- with `LLM_CACHE=true`, replies are cached in `data/llm_cache.sqlite`, so asking the same thing
  again costs no request;
- nothing here retries a rate limit: it is raised as `RateLimitedError` for the caller to report.
"""

import json
import re
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import openai
from langchain_core.caches import BaseCache
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from pydantic import BaseModel, SecretStr, ValidationError

from app.agent.prompts import JSON_ONLY_INSTRUCTIONS, JSON_RETRY_PROMPT
from app.config import Settings, get_settings

ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = ROOT / "data" / "llm_cache.sqlite"
EMBEDDING_CACHE_DIR = ROOT / "data" / "fastembed"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
LOCAL_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 384  # the width of the vector columns; both providers must match it

REQUEST_TIMEOUT_SECONDS = 60
# Providers whose models reliably support tool calling, so structured output can be native.
NATIVE_STRUCTURED_PROVIDERS = frozenset({"openai", "gemini"})

# USD per million tokens (input, output), for the cost estimate of paid models. These are list
# prices noted when this was written and are only as good as that; check before quoting a cost.
PRICES_PER_MILLION: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-2.5-flash": (0.30, 2.50),
}


class LLMError(Exception):
    """A model call failed. The message is safe to show: it never contains a key."""


class LLMConfigError(LLMError):
    """A setting needed for the chosen provider is missing."""


class RateLimitedError(LLMError):
    """The provider refused the request because of a rate or daily limit."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after  # seconds, when the provider said


class StructuredOutputError(LLMError):
    """The model did not return output that fits the schema, even after one retry."""


# --- Usage ------------------------------------------------------------------------------------


@dataclass
class Usage:
    """What a block of work cost. `requests` counts calls that reached the provider."""

    requests: int = 0
    cached: int = 0  # replies served from the local cache instead
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = 0.0  # None once a call used a paid model with no known price
    models: list[str] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def add(self, other: "Usage") -> None:
        self.requests += other.requests
        self.cached += other.cached
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        both_known = self.cost_usd is not None and other.cost_usd is not None
        self.cost_usd = self.cost_usd + other.cost_usd if both_known else None
        self.models += [name for name in other.models if name not in self.models]


_current_usage: ContextVar[Usage | None] = ContextVar("current_usage", default=None)


@contextmanager
def track_usage() -> Iterator[Usage]:
    """Collect the usage of every model call made inside the `with` block."""
    usage = Usage()
    outer = _current_usage.get()
    token = _current_usage.set(usage)
    try:
        yield usage
    finally:
        _current_usage.reset(token)
        if outer is not None:
            outer.add(usage)


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Cost in USD: 0 for `:free` models, from the price table for known ones, else None."""
    if model.endswith(":free"):
        return 0.0
    name = model.rsplit("/", 1)[-1]
    prices = PRICES_PER_MILLION.get(name)
    if prices is None:
        return None
    return (input_tokens * prices[0] + output_tokens * prices[1]) / 1_000_000


def _record(message: AIMessage, configured_model: str) -> None:
    usage = _current_usage.get()
    if usage is None:
        return
    counts: dict[str, Any] = dict(message.usage_metadata or {})
    model = str(message.response_metadata.get("model_name") or configured_model)
    # LangChain marks a reply it took from its cache with `total_cost: 0`.
    if "total_cost" in counts:
        usage.add(Usage(cached=1, models=[model]))
        return
    tokens_in, tokens_out = int(counts.get("input_tokens", 0)), int(counts.get("output_tokens", 0))
    usage.add(
        Usage(
            requests=1,
            input_tokens=tokens_in,
            output_tokens=tokens_out,
            cost_usd=estimate_cost(model, tokens_in, tokens_out),
            models=[model],
        )
    )


# --- Models -----------------------------------------------------------------------------------


@lru_cache
def _rate_limiter(requests_per_minute: int) -> InMemoryRateLimiter:
    # max_bucket_size=1: no burst, so a batch can never exceed the per-minute limit.
    return InMemoryRateLimiter(
        requests_per_second=requests_per_minute / 60, check_every_n_seconds=0.1, max_bucket_size=1
    )


@lru_cache
def _sqlite_cache(path: Path) -> BaseCache:
    from langchain_community.cache import SQLiteCache

    path.parent.mkdir(parents=True, exist_ok=True)
    return SQLiteCache(database_path=str(path))


def _key(value: SecretStr | None, variable: str, provider: str) -> SecretStr:
    if value is None or not value.get_secret_value().strip():
        raise LLMConfigError(f"{variable} is not set; LLM_PROVIDER={provider} needs it in .env")
    return value


def build_chat_model(settings: Settings) -> BaseChatModel:
    """Create the chat model the settings ask for. Nothing is sent until it is called."""
    provider = settings.llm_provider
    common: dict[str, Any] = {
        "temperature": 0,
        "max_retries": 0,  # a rate limit is reported, not looped on
        "timeout": REQUEST_TIMEOUT_SECONDS,
        "rate_limiter": _rate_limiter(settings.llm_max_rpm),
        "cache": _sqlite_cache(CACHE_PATH) if settings.llm_cache else False,
    }
    if provider == "openrouter":
        from langchain_openai import ChatOpenAI

        if not settings.llm_model:
            raise LLMConfigError("LLM_MODEL is not set; pick a model id on openrouter.ai/models")
        fallbacks = settings.fallback_models
        return ChatOpenAI(
            model=settings.llm_model,
            api_key=_key(settings.openrouter_api_key, "OPENROUTER_API_KEY", provider),
            base_url=OPENROUTER_BASE_URL,
            # OpenRouter tries these in order when the first model is busy or rate limited.
            extra_body={"models": [settings.llm_model, *fallbacks]} if fallbacks else None,
            **common,
        )
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.llm_model or DEFAULT_OPENAI_MODEL,
            api_key=_key(settings.openai_api_key, "OPENAI_API_KEY", provider),
            **common,
        )
    from langchain_google_genai import ChatGoogleGenerativeAI

    if not settings.llm_model:
        raise LLMConfigError("LLM_MODEL is not set; LLM_PROVIDER=gemini needs a model id")
    return ChatGoogleGenerativeAI(
        model=settings.llm_model,
        google_api_key=_key(settings.google_api_key, "GOOGLE_API_KEY", provider),
        **common,
    )


@lru_cache
def get_chat_model() -> BaseChatModel:
    """The process-wide chat model, from `.env`."""
    return build_chat_model(get_settings())


def build_embeddings(settings: Settings) -> Embeddings:
    """Create the embedding model the settings ask for; both give 384-dimension vectors."""
    if settings.embedding_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=OPENAI_EMBEDDING_MODEL,
            dimensions=EMBEDDING_DIMENSIONS,
            api_key=_key(settings.openai_api_key, "OPENAI_API_KEY", "openai"),
        )
    from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

    # Runs on the CPU with no key. The model (about 130 MB) is downloaded on first use.
    return FastEmbedEmbeddings(model_name=LOCAL_EMBEDDING_MODEL, cache_dir=str(EMBEDDING_CACHE_DIR))


def embedding_model_name(settings: Settings) -> str:
    """A label stored with every embedding, so vectors from two models are never mixed."""
    if settings.embedding_provider == "openai":
        return f"openai:{OPENAI_EMBEDDING_MODEL}@{EMBEDDING_DIMENSIONS}"
    return f"local:{LOCAL_EMBEDDING_MODEL}"


@lru_cache
def get_embeddings() -> Embeddings:
    """The process-wide embedding model, from `.env`."""
    return build_embeddings(get_settings())


# --- Calls ------------------------------------------------------------------------------------


def _retry_after(exc: Exception) -> float | None:
    headers = getattr(getattr(exc, "response", None), "headers", None) or {}
    try:
        return float(headers.get("retry-after", ""))
    except ValueError:
        return None


def _rate_limited(retry_after: float | None = None) -> RateLimitedError:
    when = f"Retry in {retry_after:.0f} s." if retry_after else "Wait a minute and retry;"
    return RateLimitedError(
        f"The model's per-minute or daily request limit is reached. {when} "
        "if it still fails, the daily limit is used up until tomorrow (UTC).",
        retry_after,
    )


def _is_rate_limit(exc: Exception) -> bool:
    """True for a 429 in any of the shapes the three providers use."""
    if isinstance(exc, openai.RateLimitError) or type(exc).__name__ == "ResourceExhausted":
        return True
    if getattr(exc, "status_code", None) == 429 or getattr(exc, "code", None) == 429:
        return True
    # OpenRouter can answer HTTP 200 with an error body, which LangChain raises as ValueError.
    body = exc.args[0] if isinstance(exc, ValueError) and exc.args else None
    return isinstance(body, dict) and body.get("code") == 429


def _model_name(model: BaseChatModel) -> str:
    return str(getattr(model, "model_name", None) or getattr(model, "model", "") or "unknown")


def invoke(messages: list[BaseMessage], model: BaseChatModel | None = None) -> AIMessage:
    """Send messages to the chat model, record the usage and turn failures into `LLMError`."""
    model = model or get_chat_model()
    try:
        reply = model.invoke(messages)
    except LLMError:
        raise
    except Exception as exc:
        # Each provider SDK has its own error tree; callers get one, without the SDK's text
        # (which can quote request details).
        if _is_rate_limit(exc):
            raise _rate_limited(_retry_after(exc)) from exc
        status = getattr(exc, "status_code", None)
        detail = f"HTTP {status}" if status else type(exc).__name__
        raise LLMError(f"the model request failed ({detail})") from exc
    _record(reply, _model_name(model))
    return reply


def message_text(message: BaseMessage) -> str:
    """The text of a message, whether its content is a string or a list of blocks."""
    if isinstance(message.content, str):
        return message.content
    parts = [b if isinstance(b, str) else b.get("text", "") for b in message.content]
    return "".join(str(part) for part in parts)


_THINKING = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    """A reply without the `<think>` block some free models put in front of it."""
    return _THINKING.sub("", text).strip()


def extract_json(text: str) -> dict[str, Any]:
    """Find the JSON object in a model reply that may wrap it in prose, fences or reasoning."""
    cleaned = strip_reasoning(text)
    decoder = json.JSONDecoder()
    position = cleaned.find("{")
    while position != -1:
        try:
            value, _ = decoder.raw_decode(cleaned, position)
        except json.JSONDecodeError:
            value = None
        if isinstance(value, dict):
            return value
        position = cleaned.find("{", position + 1)
    raise ValueError("the reply contains no JSON object")


def _short(error: Exception) -> str:
    if isinstance(error, ValidationError):
        problems = [
            f"{'.'.join(map(str, e['loc'])) or 'object'}: {e['msg']}" for e in error.errors()
        ]
        return "; ".join(problems[:5])
    return str(error)


def _structured_json[T: BaseModel](
    schema: type[T], messages: list[BaseMessage], model: BaseChatModel | None
) -> T:
    instructions = JSON_ONLY_INSTRUCTIONS.format(schema=json.dumps(schema.model_json_schema()))
    conversation = [*messages, HumanMessage(instructions)]
    error: Exception | None = None
    for _ in range(2):  # the first try, then one retry that shows the model its mistake
        reply = invoke(conversation, model)
        text = message_text(reply)
        try:
            return schema.model_validate(extract_json(text))
        except (ValueError, ValidationError) as exc:
            error = exc
            retry = HumanMessage(JSON_RETRY_PROMPT.format(error=_short(exc)))
            conversation = [*conversation, AIMessage(text), retry]
    raise StructuredOutputError(f"the model's reply did not fit the schema: {_short(error)}")


def _structured_native[T: BaseModel](
    schema: type[T], messages: list[BaseMessage], model: BaseChatModel
) -> T:
    runnable = model.with_structured_output(schema, include_raw=True)
    try:
        result = runnable.invoke(messages)
    except Exception as exc:
        if _is_rate_limit(exc):
            raise _rate_limited(_retry_after(exc)) from exc
        status = getattr(exc, "status_code", None)
        detail = f"HTTP {status}" if status else type(exc).__name__
        raise LLMError(f"the model request failed ({detail})") from exc
    _record(result["raw"], _model_name(model))
    parsed = result.get("parsed")
    if not isinstance(parsed, schema):
        raise StructuredOutputError("the model's reply did not fit the schema")
    return parsed


def structured[T: BaseModel](
    schema: type[T],
    messages: list[BaseMessage],
    model: BaseChatModel | None = None,
    native: bool | None = None,
) -> T:
    """Ask the model for output that fits `schema` and return the validated object.

    Paid providers use their native structured output. Free models often have no tool calling,
    so there the model is asked for JSON only, the object is pulled out of the reply and
    validated, and one retry shows the model what was wrong. `native` overrides the choice.
    """
    if native is None:
        native = get_settings().llm_provider in NATIVE_STRUCTURED_PROVIDERS
    if native:
        return _structured_native(schema, messages, model or get_chat_model())
    return _structured_json(schema, messages, model)
