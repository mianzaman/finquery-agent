"""Every prompt string, as module constants.

No prompt uses the system role: some free models reject it, and one fixed message layout keeps
replies comparable when the provider changes.
"""

# --- Structured output without tool calling (app/llm.py) ---------------------------------------

JSON_ONLY_INSTRUCTIONS = """\
Reply with one JSON object and nothing else: no explanation, no Markdown fences.
The object must match this JSON schema:
{schema}"""

JSON_RETRY_PROMPT = """\
That reply could not be used: {error}
Reply again with only the JSON object, matching the schema exactly."""

# --- Naive baseline (app/baseline.py) -----------------------------------------------------------

BASELINE_PROMPT = """\
You answer questions about US public companies by writing SQL for a PostgreSQL database.

The tables and views you can read:
{schema}

Rules:
- Write exactly one SELECT statement. Use only the tables and columns listed above.
- Companies are identified by `ticker`.
- If the question cannot be answered from this database, set "sql" to null and say why in
  "refusal". This covers investment advice, predictions, companies or data that are not in the
  database, and any request to change data.
- Otherwise put the query in "sql" and set "refusal" to null.

Question: {question}"""
