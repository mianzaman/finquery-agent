"""Tests for the model wrapper. No test reaches a provider: models are fakes or never called."""

from pathlib import Path
from typing import Any

import httpx
import openai
import pytest
from langchain_core.caches import InMemoryCache
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from app import llm
from app.config import Settings
from app.llm import (
    LLMConfigError,
    LLMError,
    RateLimitedError,
    StructuredOutputError,
    Usage,
    build_chat_model,
    build_embeddings,
    estimate_cost,
    extract_json,
    invoke,
    message_text,
    structured,
    track_usage,
)
from tests.fakes import ScriptedChat, reply


class Verdict(BaseModel):
    sql: str | None
    confidence: int


def settings(**values: Any) -> Settings:
    """Settings that ignore the developer's `.env`, with the cache off unless asked for."""
    return Settings(_env_file=None, **({"llm_cache": False} | values))


ASK = [HumanMessage("Which query?")]

# --- Cost and usage ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("some/model:free", 0.0),
        ("gpt-4o-mini", (1000 * 0.15 + 500 * 0.60) / 1_000_000),
        ("openai/gpt-4o-mini", (1000 * 0.15 + 500 * 0.60) / 1_000_000),
        ("unknown/paid-model", None),
    ],
)
def test_estimate_cost(model: str, expected: float | None) -> None:
    assert estimate_cost(model, 1000, 500) == pytest.approx(expected)


def test_usage_counts_requests_and_tokens() -> None:
    model = ScriptedChat(replies=[reply("a", 100, 20), reply("b", 50, 5)])

    with track_usage() as usage:
        invoke(ASK, model)
        invoke(ASK, model)

    assert (usage.requests, usage.cached) == (2, 0)
    assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == (150, 25, 175)
    assert usage.cost_usd == 0.0
    assert usage.models == ["test/model:free"]


def test_a_cached_reply_is_not_counted_as_a_request() -> None:
    model = ScriptedChat(replies=[reply("a", 100, 20)], cache=InMemoryCache())

    with track_usage() as usage:
        first = invoke(ASK, model)
        second = invoke(ASK, model)

    assert first.content == second.content == "a"
    assert (usage.requests, usage.cached, usage.total_tokens) == (1, 1, 120)


def test_usage_of_an_inner_block_is_added_to_the_outer_one() -> None:
    model = ScriptedChat(replies=[reply("a", 10, 1), reply("b", 10, 1)])

    with track_usage() as outer:
        invoke(ASK, model)
        with track_usage() as inner:
            invoke(ASK, model)

    assert (inner.requests, outer.requests, outer.total_tokens) == (1, 2, 22)


def test_cost_is_unknown_once_a_paid_model_has_no_price() -> None:
    usage = Usage(cost_usd=0.5)
    usage.add(Usage(requests=1, cost_usd=None))
    usage.add(Usage(requests=1, cost_usd=0.1))

    assert usage.cost_usd is None and usage.requests == 2


def test_calls_outside_a_tracking_block_still_work() -> None:
    assert invoke(ASK, ScriptedChat(replies=[reply("fine")])).content == "fine"


# --- Errors -----------------------------------------------------------------------------------


def provider_error(status: int, headers: dict[str, str] | None = None) -> openai.APIStatusError:
    request = httpx.Request("POST", "https://provider.test/v1/chat/completions")
    response = httpx.Response(status, headers=headers or {}, request=request)
    kind = openai.RateLimitError if status == 429 else openai.APIStatusError
    return kind("sk-secret-key was rejected", response=response, body=None)


class ResourceExhausted(Exception):
    """Named like the error Google's client raises for a 429."""


@pytest.mark.parametrize(
    ("error", "retry_after"),
    [
        (provider_error(429, {"retry-after": "30"}), 30.0),
        (provider_error(429), None),
        (ValueError({"message": "Rate limit exceeded: free-models-per-day", "code": 429}), None),
        (ResourceExhausted("quota"), None),
    ],
)
def test_a_rate_limit_becomes_one_clear_error(error: Exception, retry_after: float | None) -> None:
    model = ScriptedChat(replies=[error, reply("never sent")])

    with pytest.raises(RateLimitedError, match="limit is reached") as excinfo:
        invoke(ASK, model)

    assert excinfo.value.retry_after == retry_after
    assert len(model.seen) == 1  # no automatic retry


@pytest.mark.parametrize(
    ("error", "detail"),
    [
        (provider_error(401), "HTTP 401"),
        (provider_error(503), "HTTP 503"),
        (TimeoutError("sk-secret-key timed out"), "TimeoutError"),
        (ValueError({"message": "bad", "code": 400}), "ValueError"),
    ],
)
def test_other_failures_are_reported_without_the_providers_text(
    error: Exception, detail: str
) -> None:
    with pytest.raises(LLMError, match=detail) as excinfo:
        invoke(ASK, ScriptedChat(replies=[error]))

    assert not isinstance(excinfo.value, RateLimitedError)
    assert "sk-secret-key" not in str(excinfo.value)


# --- Structured output ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("<think>Is it 5?\nYes.</think>\n  It was 5.  ", "It was 5."),
        ("<THINK>a</THINK>One.<think>b</think> Two.", "One. Two."),
        ("  No reasoning here. ", "No reasoning here."),
        ("", ""),
    ],
)
def test_strip_reasoning(text: str, expected: str) -> None:
    assert llm.strip_reasoning(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        '{"sql": "SELECT 1", "confidence": 3}',
        '```json\n{"sql": "SELECT 1", "confidence": 3}\n```',
        'Here is the answer:\n{"sql": "SELECT 1", "confidence": 3}\nHope that helps.',
        '<think>maybe {"sql": 1}? no.</think>{"sql": "SELECT 1", "confidence": 3}',
        'I considered {this} first. {"sql": "SELECT 1", "confidence": 3}',
    ],
)
def test_extract_json_finds_the_object(text: str) -> None:
    assert extract_json(text) == {"sql": "SELECT 1", "confidence": 3}


@pytest.mark.parametrize("text", ["", "no json here", "[1, 2, 3]", "{broken"])
def test_extract_json_without_an_object(text: str) -> None:
    with pytest.raises(ValueError, match="no JSON object"):
        extract_json(text)


def test_message_text_joins_content_blocks() -> None:
    blocks = AIMessage(content=[{"type": "text", "text": "a"}, {"type": "text", "text": "b"}])

    assert message_text(blocks) == "ab"
    assert message_text(AIMessage(content="plain")) == "plain"


def test_structured_json_asks_for_json_and_validates_it() -> None:
    model = ScriptedChat(replies=[reply('{"sql": null, "confidence": 2}')])

    result = structured(Verdict, ASK, model=model, native=False)

    assert result == Verdict(sql=None, confidence=2)
    sent = model.seen[0]
    assert sent[0] is ASK[0]
    assert all(message.type == "human" for message in sent)  # no system role
    assert '"confidence"' in sent[-1].content and "JSON" in sent[-1].content


def test_structured_json_retries_once_with_the_validation_error() -> None:
    model = ScriptedChat(
        replies=[reply('{"sql": "SELECT 1"}'), reply('{"sql": "SELECT 1", "confidence": 9}')]
    )

    with track_usage() as usage:
        result = structured(Verdict, ASK, model=model, native=False)

    assert result.confidence == 9
    assert usage.requests == 2
    retry = model.seen[1]
    assert retry[-2].content == '{"sql": "SELECT 1"}'  # the model sees its own reply
    assert "confidence: Field required" in retry[-1].content


def test_structured_json_gives_up_after_one_retry() -> None:
    model = ScriptedChat(replies=[reply("I cannot"), reply("still prose"), reply("unused")])

    with pytest.raises(StructuredOutputError, match="no JSON object"):
        structured(Verdict, ASK, model=model, native=False)

    assert len(model.seen) == 2


def test_structured_json_does_not_retry_a_rate_limit() -> None:
    model = ScriptedChat(replies=[provider_error(429), reply("unused")])

    with pytest.raises(RateLimitedError):
        structured(Verdict, ASK, model=model, native=False)

    assert len(model.seen) == 1


class NativeModel:
    """A model whose provider supports structured output itself."""

    model_name = "gpt-4o-mini"

    def __init__(self, result: Any) -> None:
        self.result = result

    def with_structured_output(self, schema: type[BaseModel], include_raw: bool) -> "NativeModel":
        assert include_raw
        return self

    def invoke(self, messages: list[Any]) -> Any:
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def test_structured_native_returns_the_parsed_object_and_counts_usage() -> None:
    parsed = Verdict(sql="SELECT 1", confidence=5)
    model = NativeModel({"raw": reply("", 1000, 500), "parsed": parsed, "parsing_error": None})

    with track_usage() as usage:
        assert structured(Verdict, ASK, model=model, native=True) is parsed  # type: ignore[arg-type]

    assert usage.requests == 1
    assert usage.cost_usd == pytest.approx((1000 * 0.15 + 500 * 0.60) / 1_000_000)


@pytest.mark.parametrize(
    ("result", "error"),
    [
        ({"raw": reply("oops"), "parsed": None, "parsing_error": "bad"}, StructuredOutputError),
        (provider_error(429), RateLimitedError),
        (provider_error(500), LLMError),
    ],
)
def test_structured_native_failures(result: Any, error: type[Exception]) -> None:
    with pytest.raises(error):
        structured(Verdict, ASK, model=NativeModel(result), native=True)  # type: ignore[arg-type]


# --- Building models --------------------------------------------------------------------------


def test_openrouter_model_uses_the_key_base_url_and_fallbacks() -> None:
    model = build_chat_model(
        settings(
            llm_provider="openrouter",
            openrouter_api_key="sk-or-test",
            llm_model="vendor/first:free",
            llm_fallback_models="vendor/second:free, vendor/third:free",
            llm_max_rpm=12,
        )
    )

    assert type(model).__name__ == "ChatOpenAI"
    assert model.model_name == "vendor/first:free"
    assert model.openai_api_base == "https://openrouter.ai/api/v1"
    assert model.openai_api_key.get_secret_value() == "sk-or-test"
    assert model.extra_body == {
        "models": ["vendor/first:free", "vendor/second:free", "vendor/third:free"]
    }
    assert model.max_retries == 0 and model.temperature == 0
    assert model.rate_limiter.requests_per_second == pytest.approx(0.2)
    assert model.cache is False


def test_openrouter_without_fallbacks_sends_no_model_list() -> None:
    model = build_chat_model(
        settings(llm_provider="openrouter", openrouter_api_key="k", llm_model="vendor/one:free")
    )

    assert model.extra_body is None


def test_openai_model_has_a_default_and_gemini_is_built_from_its_key() -> None:
    openai_model = build_chat_model(settings(llm_provider="openai", openai_api_key="sk-test"))
    gemini = build_chat_model(
        settings(llm_provider="gemini", google_api_key="g-test", llm_model="gemini-2.0-flash")
    )

    assert openai_model.model_name == "gpt-4o-mini"
    assert openai_model.openai_api_base is None  # OpenAI itself, not OpenRouter
    assert type(gemini).__name__ == "ChatGoogleGenerativeAI"
    assert gemini.model.endswith("gemini-2.0-flash")


@pytest.mark.parametrize(
    ("values", "variable"),
    [
        ({"llm_provider": "openrouter", "llm_model": "vendor/one:free"}, "OPENROUTER_API_KEY"),
        ({"llm_provider": "openrouter", "openrouter_api_key": " ", "llm_model": "m"}, "OPENROUTER"),
        ({"llm_provider": "openrouter", "openrouter_api_key": "k"}, "LLM_MODEL"),
        ({"llm_provider": "openai"}, "OPENAI_API_KEY"),
        ({"llm_provider": "gemini", "llm_model": "gemini-2.0-flash"}, "GOOGLE_API_KEY"),
        ({"llm_provider": "gemini", "google_api_key": "g"}, "LLM_MODEL"),
    ],
)
def test_a_missing_setting_is_named(values: dict[str, Any], variable: str) -> None:
    with pytest.raises(LLMConfigError, match=variable):
        build_chat_model(settings(**values))


def test_the_cache_is_a_sqlite_file_when_switched_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(llm, "CACHE_PATH", tmp_path / "cache" / "llm.sqlite")

    model = build_chat_model(
        settings(llm_provider="openai", openai_api_key="sk-test", llm_cache=True)
    )

    assert type(model.cache).__name__ == "SQLiteCache"
    assert (tmp_path / "cache" / "llm.sqlite").exists()


def test_local_embeddings_use_the_small_bge_model(monkeypatch: pytest.MonkeyPatch) -> None:
    created: dict[str, Any] = {}

    class FakeFastEmbed:
        def __init__(self, **arguments: Any) -> None:
            created.update(arguments)

    monkeypatch.setattr(
        "langchain_community.embeddings.fastembed.FastEmbedEmbeddings", FakeFastEmbed
    )

    assert isinstance(build_embeddings(settings(embedding_provider="local")), FakeFastEmbed)
    assert created["model_name"] == "BAAI/bge-small-en-v1.5"
    assert created["cache_dir"].endswith("data/fastembed")


def test_openai_embeddings_are_cut_to_the_same_384_dimensions() -> None:
    embeddings = build_embeddings(settings(embedding_provider="openai", openai_api_key="sk-test"))

    assert (embeddings.model, embeddings.dimensions) == ("text-embedding-3-small", 384)
    with pytest.raises(LLMConfigError, match="OPENAI_API_KEY"):
        build_embeddings(settings(embedding_provider="openai"))
