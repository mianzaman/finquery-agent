"""Tests for `GET /schema`: the owner's descriptions, read from `db/schema_docs.yaml`."""

from fastapi.testclient import TestClient

from app import service
from app.agent.nodes.validate_sql import ALLOWED_TABLES
from app.main import create_app


def test_schema_lists_the_tables_and_columns_with_descriptions() -> None:
    response = TestClient(create_app()).get("/schema")

    assert response.status_code == 200
    tables = {table["name"]: table for table in response.json()["tables"]}
    assert "companies" in tables and "daily_prices" in tables
    assert tables["companies"]["description"].startswith("One row per company")
    columns = {column["name"]: column["description"] for column in tables["companies"]["columns"]}
    assert columns["ticker"].startswith("Stock ticker symbol")
    assert all(table["description"] and table["columns"] for table in tables.values())
    assert response.headers["x-request-id"]


def test_schema_shows_only_tables_the_agent_may_query() -> None:
    names = {table.name for table in service.schema().tables}

    assert names <= set(ALLOWED_TABLES)
