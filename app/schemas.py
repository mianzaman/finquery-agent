"""Pydantic request and response models: the API contract."""

from typing import Annotated, Any

from pydantic import BaseModel, Field, StringConstraints

QUESTION_MIN_LENGTH = 3
QUESTION_MAX_LENGTH = 500

Question = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True, min_length=QUESTION_MIN_LENGTH, max_length=QUESTION_MAX_LENGTH
    ),
]


class AskRequest(BaseModel):
    """A question in plain English."""

    question: Question


class AskResponse(BaseModel):
    """What the agent made of a question. A refusal is a normal response, not an error."""

    answer: str | None = Field(description="1 to 3 sentences; null when the question is refused")
    sql: str | None = Field(
        description="The query behind the rows. When the answer says the query still failed, "
        "this is the last attempt, which returned nothing."
    )
    columns: list[str]
    rows: list[list[Any]] = Field(description="At most 500 rows; dates and decimals as JSON")
    refused: bool
    refusal_reason: str | None
    attempts: int = Field(description="Repairs the query needed (0 to 2)")
    faithfulness_ok: bool | None = Field(
        description="True: every number in the answer is in the rows. False: the answer was "
        "replaced by a neutral sentence. Null: there was no model-written answer to check."
    )
    latency_ms: int
    request_id: str


class ErrorResponse(BaseModel):
    """Every failure, in one shape. `message` is safe to show to the user."""

    code: str
    message: str
    request_id: str
    retry_after: int | None = Field(default=None, description="Seconds to wait, when known")


class ColumnDoc(BaseModel):
    name: str
    description: str


class TableDoc(BaseModel):
    name: str
    description: str
    columns: list[ColumnDoc]


class SchemaResponse(BaseModel):
    """The tables the agent can query, as described in `db/schema_docs.yaml`."""

    tables: list[TableDoc]
