"""Tests for the settings object."""

import pytest

from app.config import Settings

ENV_VARS = [
    "LLM_PROVIDER",
    "OPENROUTER_API_KEY",
    "LLM_MODEL",
    "LLM_FALLBACK_MODELS",
    "OPENAI_API_KEY",
    "GOOGLE_API_KEY",
    "OLLAMA_BASE_URL",
    "EMBEDDING_PROVIDER",
    "LLM_MAX_RPM",
    "LLM_CACHE",
    "SEC_USER_AGENT",
    "DATABASE_URL",
    "READER_DATABASE_URL",
]


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Remove every project variable so a test never sees the developer's shell or `.env`."""
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_defaults_without_env(clean_env: pytest.MonkeyPatch) -> None:
    settings = Settings(_env_file=None)

    assert settings.llm_provider == "openrouter"
    assert settings.embedding_provider == "local"
    assert settings.llm_max_rpm == 15
    assert settings.llm_cache is True
    assert settings.openrouter_api_key is None
    assert settings.fallback_models == []
    assert settings.ollama_base_url == "http://localhost:11434"


def test_reads_environment(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("LLM_PROVIDER", "openai")
    clean_env.setenv("LLM_MAX_RPM", "5")
    clean_env.setenv("OPENAI_API_KEY", "test-key")

    settings = Settings(_env_file=None)

    assert settings.llm_provider == "openai"
    assert settings.llm_max_rpm == 5
    assert settings.openai_api_key is not None
    assert settings.openai_api_key.get_secret_value() == "test-key"


def test_ollama_is_a_provider_with_its_own_address(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("LLM_PROVIDER", "ollama")
    clean_env.setenv("LLM_MODEL", "some-model:3b")
    clean_env.setenv("OLLAMA_BASE_URL", "http://gpu-box:11434")

    settings = Settings(_env_file=None)

    assert (settings.llm_provider, settings.llm_model) == ("ollama", "some-model:3b")
    assert settings.ollama_base_url == "http://gpu-box:11434"


@pytest.mark.parametrize("provider", ["openrouter", "openai", "gemini", "ollama"])
def test_every_documented_provider_is_accepted(
    clean_env: pytest.MonkeyPatch, provider: str
) -> None:
    clean_env.setenv("LLM_PROVIDER", provider)

    assert Settings(_env_file=None).llm_provider == provider


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", []),
        ("a:free", ["a:free"]),
        ("a:free, b:free", ["a:free", "b:free"]),
        (" a:free ,, ", ["a:free"]),
    ],
)
def test_fallback_models_parsing(
    clean_env: pytest.MonkeyPatch, raw: str, expected: list[str]
) -> None:
    clean_env.setenv("LLM_FALLBACK_MODELS", raw)

    assert Settings(_env_file=None).fallback_models == expected


def test_secrets_are_masked_in_repr(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("OPENROUTER_API_KEY", "super-secret")

    assert "super-secret" not in repr(Settings(_env_file=None))


def test_unknown_provider_is_rejected(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("LLM_PROVIDER", "nope")

    with pytest.raises(ValueError):
        Settings(_env_file=None)
