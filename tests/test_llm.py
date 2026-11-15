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


# --- Ollama (a local model): the class is replaced, so nothing is sent --------------------------


class ChatOllama:
    """Stands in for `langchain_ollama.ChatOllama` (the wrapper recognises it by this name).

    `native` is what the JSON-schema call returns (or raises); `replies` answer plain calls.
    """

    def __init__(self, native: Any = None, replies: list[Any] | None = None, **kwargs: Any) -> None:
        self.native = native
        self.replies = list(replies or [])
        self.kwargs = kwargs
        self.model = kwargs.get("model", "local-model:3b")
        self.base_url = kwargs.get("base_url", "http://localhost:11434")
        self.methods: list[str] = []
        self.seen: list[list[Any]] = []

    def with_structured_output(
        self, schema: type[BaseModel], include_raw: bool, method: str
    ) -> "ChatOllama":
        assert include_raw
        self.methods.append(method)
        return _OllamaStructured(self)  # type: ignore[return-value]

    def invoke(self, messages: list[Any]) -> Any:
        self.seen.append(list(messages))
        item = self.replies.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class _OllamaStructured:
    def __init__(self, model: ChatOllama) -> None:
        self.model = model

    def invoke(self, messages: list[Any]) -> Any:
        if isinstance(self.model.native, Exception):
            raise self.model.native
        return self.model.native


@pytest.fixture
def ollama_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """`LLM_PROVIDER=ollama`, so `structured()` picks the Ollama path by itself."""
    chosen = settings(llm_provider="ollama", llm_model="local-model:3b")
    monkeypatch.setattr(llm, "get_settings", lambda: chosen)


def test_ollama_model_is_local_uncapped_and_patient(monkeypatch: pytest.MonkeyPatch) -> None:
    import langchain_ollama

    monkeypatch.setattr(langchain_ollama, "ChatOllama", ChatOllama)

    model = build_chat_model(
        settings(
            llm_provider="ollama", llm_model="local-model:3b", ollama_base_url="http://box:11434"
        )
    )

    assert isinstance(model, ChatOllama)
    assert model.kwargs == {
        "model": "local-model:3b",
        "base_url": "http://box:11434",
        "temperature": 0,
        "client_kwargs": {"timeout": llm.OLLAMA_TIMEOUT_SECONDS},
        "cache": False,
    }
    assert "rate_limiter" not in model.kwargs  # a local server has no request limit
    assert llm.OLLAMA_TIMEOUT_SECONDS > llm.REQUEST_TIMEOUT_SECONDS


def test_ollama_keeps_the_reply_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import langchain_ollama

    monkeypatch.setattr(langchain_ollama, "ChatOllama", ChatOllama)
    monkeypatch.setattr(llm, "CACHE_PATH", tmp_path / "cache.sqlite")

    model = build_chat_model(settings(llm_provider="ollama", llm_model="m", llm_cache=True))

    assert type(model.kwargs["cache"]).__name__ == "SQLiteCache"  # type: ignore[attr-defined]


def test_ollama_needs_a_model_name_but_no_key() -> None:
    with pytest.raises(LLMConfigError, match="LLM_MODEL is not set; LLM_PROVIDER=ollama"):
        build_chat_model(settings(llm_provider="ollama"))


def test_the_real_ollama_class_is_built_without_a_request() -> None:
    """Constructing the real class must not contact a server (none runs in CI)."""
    model = build_chat_model(
        settings(llm_provider="ollama", llm_model="m", ollama_base_url="http://127.0.0.1:9")
    )

    assert type(model).__name__ == "ChatOllama"
    assert (model.model, model.base_url, model.temperature) == ("m", "http://127.0.0.1:9", 0)  # type: ignore[attr-defined]
    assert model.rate_limiter is None


def test_ollama_structured_output_uses_the_json_schema_and_costs_nothing(
    ollama_settings: None,
) -> None:
    parsed = Verdict(sql="SELECT 1", confidence=5)
    model = ChatOllama(native={"raw": reply("", 1000, 500), "parsed": parsed})

    with track_usage() as usage:
        assert structured(Verdict, ASK, model=model) is parsed  # type: ignore[arg-type]

    assert model.methods == ["json_schema"]
    assert model.seen == []  # no second request
    assert (usage.requests, usage.total_tokens, usage.cost_usd) == (1, 1500, 0.0)


def test_ollama_falls_back_to_parsing_json_with_one_retry(ollama_settings: None) -> None:
    model = ChatOllama(
        native={"raw": reply('{"sql": "SELECT'), "parsed": None, "parsing_error": "cut off"},
        replies=[reply("not json"), reply('{"sql": "SELECT 1", "confidence": 4}')],
    )

    with track_usage() as usage:
        result = structured(Verdict, ASK, model=model)  # type: ignore[arg-type]

    assert result == Verdict(sql="SELECT 1", confidence=4)
    assert len(model.seen) == 2  # the plain request and its one retry
    assert "could not be used" in model.seen[1][-1].content
    assert (usage.requests, usage.cost_usd) == (3, 0.0)


def test_ollama_gives_up_after_the_fallback_and_its_retry(ollama_settings: None) -> None:
    model = ChatOllama(
        native={"raw": reply("x"), "parsed": None},
        replies=[reply("no"), reply("still no"), reply("never asked")],
    )

    with pytest.raises(StructuredOutputError):
        structured(Verdict, ASK, model=model)  # type: ignore[arg-type]

    assert len(model.seen) == 2


def status_error(status: int) -> Exception:
    error = Exception("provider text that must not be shown")
    error.status_code = status  # type: ignore[attr-defined]
    return error


@pytest.mark.parametrize(
    ("failure", "kind", "message"),
    [
        (
            httpx.ConnectError("[Errno 111] Connection refused"),
            LLMConfigError,
            "Ollama is not running at http://localhost:11434. Start Ollama: `ollama serve`",
        ),
        (
            ConnectionError("Failed to connect to Ollama"),
            LLMConfigError,
            "Ollama is not running at http://localhost:11434. Start Ollama: `ollama serve`",
        ),
        (
            status_error(404),
            LLMConfigError,
            "Ollama has no model named local-model:3b. Pull it: `ollama pull local-model:3b`",
        ),
        (
            httpx.ReadTimeout("timed out"),
            LLMError,
            "the local model did not answer in 300 s; try a smaller model",
        ),
        (status_error(500), LLMError, "the model request failed (HTTP 500)"),
    ],
)
def test_ollama_failures_say_what_to_do(
    ollama_settings: None, failure: Exception, kind: type[Exception], message: str
) -> None:
    structured_model = ChatOllama(native=failure)
    plain_model = ChatOllama(replies=[failure])

    with pytest.raises(kind) as from_structured:
        structured(Verdict, ASK, model=structured_model)  # type: ignore[arg-type]
    with pytest.raises(kind) as from_plain:
        llm.invoke(ASK, model=plain_model)  # type: ignore[arg-type]

    assert str(from_structured.value) == str(from_plain.value) == message
    assert structured_model.seen == []  # a failed request is not retried the plain way


def test_a_connection_error_from_another_provider_does_not_mention_ollama() -> None:
    model = ScriptedChat(replies=[httpx.ConnectError("refused")])

    with pytest.raises(LLMError, match=r"the model request failed \(ConnectError\)"):
        llm.invoke(ASK, model=model)


def test_the_other_providers_keep_their_structured_output_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for provider, native in (("openrouter", False), ("openai", True)):
        chosen = settings(llm_provider=provider)
        monkeypatch.setattr(llm, "get_settings", lambda chosen=chosen: chosen)
        parsed = Verdict(sql=None, confidence=1)
        model = NativeModel({"raw": reply("", 1, 1), "parsed": parsed})
        if native:
            assert structured(Verdict, ASK, model=model) is parsed  # type: ignore[arg-type]
        else:
            scripted = ScriptedChat(replies=[reply('{"sql": null, "confidence": 1}')])
            assert structured(Verdict, ASK, model=scripted) == parsed
