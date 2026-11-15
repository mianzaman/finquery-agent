"""Streamlit UI: a chat page that talks to the API over HTTP only.

Run with `uv run streamlit run ui/streamlit_app.py` (the API must be running; its address is
`API_URL`). Each answer shows the text, the SQL in an expander, the rows as a table and a
caption with the latency and the repairs. Nothing here imports the agent.
"""

import sys
from pathlib import Path
from typing import Any

import httpx
import streamlit as st

# `streamlit run ui/streamlit_app.py` puts `ui/` on the import path, not the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402  (must follow the path line above)

TIMEOUT_SECONDS = 120  # free models are slow, and a question can need several calls
EXAMPLES = (
    "What was Apple's revenue last quarter?",
    "How has Microsoft's net income changed over its last 6 quarters?",
    "Compare Coca-Cola's and PepsiCo's operating income in fiscal 2025.",
    "Which 5 companies had the most cash at the end of their latest quarter?",
    "What was Nvidia's highest closing share price in 2025?",
    "Which Energy company paid the most dividends per share in calendar 2025?",
)
UNREACHABLE = "The API could not be reached. Is it running?"
TIMED_OUT = "The API took too long to answer. Try again in a moment."
UNREADABLE = "The API sent a reply this page could not read."


def ask_api(question: str, api_url: str) -> dict[str, Any]:
    """Send the question to `POST /ask`. Returns the response body, or `{"error": message}`."""
    try:
        response = httpx.post(
            f"{api_url.rstrip('/')}/ask", json={"question": question}, timeout=TIMEOUT_SECONDS
        )
    except httpx.TimeoutException:
        return {"error": TIMED_OUT}
    except httpx.HTTPError:
        return {"error": UNREACHABLE}
    try:
        body = response.json()
    except ValueError:
        body = None
    if not isinstance(body, dict):
        return {"error": UNREADABLE}
    if response.status_code != 200:
        # The API's own errors carry a message written for the user.
        return {
            "error": str(body.get("message") or UNREADABLE),
            "request_id": body.get("request_id"),
        }
    return body


def caption(result: dict[str, Any]) -> str:
    """`2.3 s · 1 repair · numbers checked against the rows`."""
    parts = [f"{result.get('latency_ms', 0) / 1000:.1f} s"]
    attempts = result.get("attempts", 0)
    parts.append(
        "no repairs" if not attempts else f"{attempts} repair{'s' if attempts > 1 else ''}"
    )
    checked = result.get("faithfulness_ok")
    if checked is True:
        parts.append("numbers checked against the rows")
    elif checked is False:
        parts.append("the written answer was withheld: its numbers did not match the rows")
    return " · ".join(parts)


def render(result: dict[str, Any]) -> None:
    """Show one API result inside the current chat message."""
    if result.get("error"):
        st.error(result["error"])
        return
    if result.get("refused"):
        st.info(result.get("refusal_reason") or "That question is out of scope.")
        return
    st.markdown(result.get("answer") or "")
    if result.get("sql"):
        with st.expander("SQL"):
            st.code(result["sql"], language="sql")
    if result.get("rows"):
        st.dataframe([dict(zip(result["columns"], row, strict=False)) for row in result["rows"]])
    st.caption(caption(result))


def main() -> None:
    st.set_page_config(page_title="FinQuery Agent", page_icon="📊")
    st.title("FinQuery Agent")
    st.caption(
        "Ask about 50 large US companies: quarterly financials from SEC filings and daily "
        "share prices, 2021 to 2026. Answers come from SQL you can read. Not investment advice."
    )
    history: list[dict[str, Any]] = st.session_state.setdefault("history", [])

    picked: str | None = None
    columns = st.columns(2)
    for index, example in enumerate(EXAMPLES):
        if columns[index % 2].button(example, key=f"example_{index}", use_container_width=True):
            picked = example
    question = st.chat_input("Ask a question about a company") or picked

    for entry in history:
        with st.chat_message("user"):
            st.markdown(entry["question"])
        with st.chat_message("assistant"):
            render(entry["result"])

    if question:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Writing and running the query…"):
                result = ask_api(question, get_settings().api_url)
            render(result)
        history.append({"question": question, "result": result})


main()
