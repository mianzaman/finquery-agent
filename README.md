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

## Switching models

The model is set in `.env`. Pick one block:

```bash
# Local, free, no request limit (pull the model first: `ollama pull qwen2.5-coder:3b`)
LLM_PROVIDER=ollama
LLM_MODEL=qwen2.5-coder:3b
OLLAMA_BASE_URL=http://localhost:11434   # in docker compose: http://host.docker.internal:11434

# OpenRouter free models (50 requests a day)
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=...
LLM_MODEL=<model id ending in :free>
LLM_FALLBACK_MODELS=<optional, comma-separated :free ids>

# OpenAI
LLM_PROVIDER=openai
OPENAI_API_KEY=...
LLM_MODEL=<OpenAI model id>

# Gemini
LLM_PROVIDER=gemini
GOOGLE_API_KEY=...
LLM_MODEL=<Gemini model id>
```

For one eval run, flags override `.env` without editing it (they also drop
`LLM_FALLBACK_MODELS`):

```bash
uv run python -m evals.run_eval --system agent --provider ollama --model qwen2.5-coder:3b
uv run python -m evals.run_eval --system agent --model <another model of the .env provider>
```

Every results file records the provider and the model. `--resume` only continues a run made
with the same model, and `--compare` refuses runs made with different models unless you pass
`--allow-mixed`. Results from a small local model are for checking the pipeline; do not compare
them with a hosted model's.
