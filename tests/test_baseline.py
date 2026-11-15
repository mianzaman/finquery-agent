"""Tests for the naive baseline. The model is a fake; the database is the local one."""

import openai
import pytest
from pydantic import BaseModel

from app import baseline, llm
from app.baseline import answer, schema_text
from app.llm import RateLimitedError
from tests.fakes import ScriptedChat, reply
from tests.test_llm import provider_error

pytestmark = pytest.mark.usefixtures("db")


@pytest.fixture
def model(monkeypatch: pytest.MonkeyPatch) -> ScriptedChat:
    """A scripted model wired into the baseline through the real JSON path of `structured`."""
    fake = ScriptedChat(replies=[])

    def structured(schema: type[BaseModel], messages: list) -> BaseModel:
        return llm.structured(schema, messages, model=fake, native=False)

    monkeypatch.setattr(baseline, "structured", structured)
    return fake


def test_schema_lists_what_the_reader_can_query() -> None:
    schema = schema_text()

    for table in ("companies", "income_statements", "daily_prices", "dividends", "stock_splits"):
        assert f"\n{table}(" in f"\n{schema}"
    assert "income_statements_split_adjusted(" in schema
    assert "eps_diluted_adjusted numeric" in schema
    for hidden in ("schema_migrations", "schema_docs", "query_examples", "pg_"):
        assert hidden not in schema


def test_a_question_becomes_sql_and_rows(model: ScriptedChat) -> None:
    sql = "SELECT n, n * 2 AS doubled FROM unnest(ARRAY[1, 2, 3]) AS n ORDER BY n"
    model.replies = [reply(f'{{"sql": "{sql}", "refusal": null}}', 900, 40)]

    result = answer("Count to three and double it")

    assert result["sql"] == sql
    assert (result["columns"], result["rows"]) == (["n", "doubled"], [(1, 2), (2, 4), (3, 6)])
    assert (result["error"], result["refusal"], result["truncated"]) == (None, None, False)
    assert (result["tokens"], result["requests"], result["cost_usd"]) == (940, 1, 0.0)
    assert result["cached"] == 0
    assert result["latency_ms"] >= 0
    prompt = model.seen[0][0].content
    assert "Count to three and double it" in prompt and "daily_prices(" in prompt


def test_a_refusal_runs_no_sql(model: ScriptedChat) -> None:
    model.replies = [reply('{"sql": null, "refusal": "That is investment advice."}')]

    result = answer("Should I buy Nvidia?")

    assert (result["sql"], result["rows"], result["error"]) == (None, [], None)
    assert result["refusal"] == "That is investment advice."


def test_no_sql_and_no_reason_still_counts_as_a_refusal(model: ScriptedChat) -> None:
    model.replies = [reply('{"sql": "  ", "refusal": null}')]

    assert answer("?")["refusal"] == "the model returned no SQL"


@pytest.mark.parametrize(
    ("sql", "error"),
    [
        ("DELETE FROM companies", "rejected: only SELECT"),
        ("SELECT set_config('statement_timeout', '0', false)", "rejected: the function set_config"),
        ("SELECT 1; SELECT 2", "rejected: 2 statements"),
        ("SELECT nope FROM companies", "database: column"),
    ],
)
def test_bad_sql_is_reported_and_never_harms(model: ScriptedChat, sql: str, error: str) -> None:
    model.replies = [reply(f'{{"sql": "{sql}", "refusal": null}}')]

    result = answer("anything")

    assert result["sql"] == sql
    assert result["error"].startswith(error)
    assert result["rows"] == [] and result["refusal"] is None


def test_a_model_failure_is_an_error_not_a_refusal(model: ScriptedChat) -> None:
    model.replies = [provider_error(503)]

    result = answer("anything")

    assert result["error"] == "model: the model request failed (HTTP 503)"
    assert (result["sql"], result["refusal"]) == (None, None)


def test_unusable_output_costs_two_requests_and_is_an_error(model: ScriptedChat) -> None:
    model.replies = [reply("no json"), reply("still none")]

    result = answer("anything")

    assert result["error"].startswith("model: the model's reply did not fit the schema")
    assert result["requests"] == 2


def test_a_rate_limit_is_raised_so_a_batch_can_stop(model: ScriptedChat) -> None:
    model.replies = [provider_error(429)]

    with pytest.raises(RateLimitedError):
        answer("anything")
    assert isinstance(provider_error(429), openai.RateLimitError)
