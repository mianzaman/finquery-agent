"""Application settings: the only place environment variables are read."""

from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Every setting the project reads from the environment or `.env`."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    llm_provider: Literal["openrouter", "openai", "gemini"] = "openrouter"
    openrouter_api_key: SecretStr | None = None
    llm_model: str = ""
    llm_fallback_models: str = ""
    openai_api_key: SecretStr | None = None
    google_api_key: SecretStr | None = None

    # Embeddings
    embedding_provider: Literal["local", "openai"] = "local"

    # Request budget
    llm_max_rpm: int = 15
    llm_cache: bool = True

    # Ingestion
    sec_user_agent: str = ""

    # Database (SecretStr because the URLs contain passwords)
    database_url: SecretStr = SecretStr(
        "postgresql+psycopg://finquery:finquery@localhost:5433/finquery"
    )
    reader_database_url: SecretStr = SecretStr(
        "postgresql+psycopg://agent_reader:reader_pw@localhost:5433/finquery"
    )

    @property
    def fallback_models(self) -> list[str]:
        """`LLM_FALLBACK_MODELS` split on commas, without blanks."""
        return [m.strip() for m in self.llm_fallback_models.split(",") if m.strip()]


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings object."""
    return Settings()
