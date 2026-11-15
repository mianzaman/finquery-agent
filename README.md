# FinQuery Agent

Open-source AI financial analyst. Ask a plain-English question about a public US company and a
LangGraph agent writes, validates and runs SQL over SEC EDGAR and yfinance data in PostgreSQL.

Work in progress. The full README is written in Task 13.

## Quick start

```bash
cp .env.example .env          # then fill in your keys
docker compose up -d db
uv sync
uv run pytest -q
uv run uvicorn app.main:app --reload
```
