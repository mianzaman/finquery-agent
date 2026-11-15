"""LangGraph wiring: nodes, edges and routing functions.

    guardrail -> (refused: END) -> intent -> entities -> retrieve_schema -> generate_sql
        -> validate_sql -> execute -> answer -> END

A query that fails validation or execution goes to repair_sql and back to validate_sql, at most
`MAX_REPAIRS` times; after that the answer node reports the failure. The limit is checked before
a repair is made, so every repaired query is validated and none is thrown away unused.

No node logic lives here: routing functions only read the state.
"""

from collections.abc import Callable
from functools import lru_cache
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agent.nodes.answer import answer
from app.agent.nodes.entities import entities
from app.agent.nodes.execute import execute
from app.agent.nodes.generate_sql import generate_sql
from app.agent.nodes.guardrail import guardrail
from app.agent.nodes.intent import intent
from app.agent.nodes.repair_sql import repair_sql
from app.agent.nodes.retrieve_schema import retrieve_schema
from app.agent.nodes.validate_sql import validate_sql
from app.agent.state import AgentState

MAX_REPAIRS = 2

Node = Callable[[AgentState], dict[str, Any]]

NODES: dict[str, Node] = {
    "guardrail": guardrail,
    "intent": intent,
    "entities": entities,
    "retrieve_schema": retrieve_schema,
    "generate_sql": generate_sql,
    "validate_sql": validate_sql,
    "execute": execute,
    "repair_sql": repair_sql,
    "answer": answer,
}


def route_after_guardrail(state: AgentState) -> str:
    """A refused question ends here; anything else goes on to intent."""
    return END if state.get("refused") else "intent"


def _repair_or_give_up(state: AgentState) -> str:
    return "repair_sql" if state.get("attempts", 0) < MAX_REPAIRS else "answer"


def route_after_validation(state: AgentState) -> str:
    """Valid SQL is executed; invalid SQL is repaired, until the repairs run out."""
    return _repair_or_give_up(state) if state.get("validation_error") else "execute"


def route_after_execute(state: AgentState) -> str:
    """Rows go to the answer; a database error is repaired, until the repairs run out."""
    return _repair_or_give_up(state) if state.get("db_error") else "answer"


def build_graph(overrides: dict[str, Node] | None = None) -> CompiledStateGraph:
    """Wire the agent. `overrides` replaces nodes by name; it exists for tests, to force a route.

    Application code calls `run_agent`, which never passes overrides.
    """
    unknown = set(overrides or {}) - set(NODES)
    if unknown:
        raise ValueError(f"no such nodes: {', '.join(sorted(unknown))}")
    graph = StateGraph(AgentState)
    for name, node in (NODES | (overrides or {})).items():
        graph.add_node(name, node)

    graph.add_edge(START, "guardrail")
    graph.add_conditional_edges("guardrail", route_after_guardrail, ["intent", END])
    graph.add_edge("intent", "entities")
    graph.add_edge("entities", "retrieve_schema")
    graph.add_edge("retrieve_schema", "generate_sql")
    graph.add_edge("generate_sql", "validate_sql")
    graph.add_conditional_edges(
        "validate_sql", route_after_validation, ["execute", "repair_sql", "answer"]
    )
    graph.add_conditional_edges("execute", route_after_execute, ["answer", "repair_sql"])
    graph.add_edge("repair_sql", "validate_sql")
    graph.add_edge("answer", END)
    return graph.compile()


@lru_cache
def _default_graph() -> CompiledStateGraph:
    return build_graph()


def initial_state(question: str) -> AgentState:
    """The state a run starts from."""
    return {"question": question, "attempts": 0, "trace": []}


def run_agent(question: str) -> AgentState:
    """Run one question through the graph and return the final state.

    It always uses the graph with the real nodes, so no caller can swap out `validate_sql`.
    """
    return _default_graph().invoke(initial_state(question))  # type: ignore[return-value]
