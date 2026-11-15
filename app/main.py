"""FastAPI app and routes. The work is in `app/service.py`; this file is wiring.

Every response carries an `X-Request-ID`, and every failure is an `ErrorResponse` with that
id: no stack trace, SQL error or connection detail reaches the client. One JSON log line is
written per request.
"""

import logging
import threading
import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import Depends, FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app import service
from app.config import get_settings
from app.llm import LLMConfigError, LLMError, RateLimitedError, StructuredOutputError
from app.log import configure_logging
from app.rate_limit import RateLimiter
from app.schemas import AskRequest, AskResponse, ErrorResponse, SchemaResponse

logger = logging.getLogger(__name__)

MODEL_RETRY_SECONDS = 60  # suggested wait when the model's limit gives no time of its own
BUSY_RETRY_SECONDS = 5
MAX_BODY_BYTES = 10_000  # a 500-character question is at most about 2 kB of JSON
ERRORS = {code: {"model": ErrorResponse} for code in (413, 422, 429, 500, 502, 503)}
HTTP_ERROR_CODES = {400: "invalid_request", 404: "not_found", 405: "method_not_allowed"}


class Busy(Exception):
    """As many questions as the server allows are already being answered."""


class TooManyRequests(Exception):
    """One client sent more /ask requests than the limit allows."""

    def __init__(self, retry_after: int) -> None:
        super().__init__("too many requests")
        self.retry_after = retry_after


def error_response(
    request: Request, status: int, code: str, message: str, retry_after: int | None = None
) -> JSONResponse:
    """An `ErrorResponse` with the request's id, and `Retry-After` when there is a wait."""
    body = ErrorResponse(
        code=code,
        message=message,
        request_id=getattr(request.state, "request_id", "unknown"),
        retry_after=retry_after,
    )
    headers = {"Retry-After": str(retry_after)} if retry_after else None
    return JSONResponse(body.model_dump(), status_code=status, headers=headers)


def _declared_length(request: Request) -> int:
    """The body size the client announced; 0 when it gave none or something unreadable."""
    try:
        return int(request.headers.get("content-length", "0"))
    except ValueError:
        return 0


def create_app(rate_limit: int | None = None, max_concurrent: int | None = None) -> FastAPI:
    """Build the API. The arguments override the settings of the same purpose; tests use them."""
    settings = get_settings()
    configure_logging()
    api = FastAPI(title="FinQuery Agent")
    limiter = RateLimiter(rate_limit or settings.api_rate_limit_per_minute)
    # Bounds the model calls and database queries in flight, whatever the number of clients.
    slots = threading.BoundedSemaphore(max_concurrent or settings.api_max_concurrent)

    @api.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Give the request an id, turn an unexpected failure into a 500, and log one line."""
        request.state.request_id = uuid.uuid4().hex
        started = time.perf_counter()
        try:
            if _declared_length(request) > MAX_BODY_BYTES:
                response = error_response(
                    request, 413, "request_too_large", "The request body is too large."
                )
            else:
                response = await call_next(request)
        except Exception:
            # The detail goes to the log (the exception's type only); the client gets none.
            logger.exception("unhandled error", extra={"request_id": request.state.request_id})
            response = error_response(
                request, 500, "internal_error", "Something went wrong on the server."
            )
        response.headers["X-Request-ID"] = request.state.request_id
        fields = {
            "request_id": request.state.request_id,
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "latency_ms": round((time.perf_counter() - started) * 1000),
        }
        logger.info("request", extra=fields | getattr(request.state, "log_fields", {}))
        return response

    # Added after the request middleware, so it wraps it: error responses get CORS headers too.
    api.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
        expose_headers=["X-Request-ID", "Retry-After"],
    )

    @api.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        problems = [
            f"{'.'.join(str(part) for part in error['loc'][1:]) or 'body'}: {error['msg']}"
            for error in exc.errors()
        ]
        return error_response(request, 422, "invalid_request", "; ".join(problems[:3]))

    @api.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        """Unknown paths, wrong methods and unreadable bodies, in the same shape as the rest."""
        code = HTTP_ERROR_CODES.get(exc.status_code, "http_error")
        return error_response(request, exc.status_code, code, str(exc.detail))

    @api.exception_handler(Busy)
    async def busy(request: Request, exc: Busy) -> JSONResponse:
        message = "The server is answering as many questions as it can. Try again in a moment."
        return error_response(request, 503, "busy", message, BUSY_RETRY_SECONDS)

    @api.exception_handler(TooManyRequests)
    async def too_many_requests(request: Request, exc: TooManyRequests) -> JSONResponse:
        message = f"Too many questions from this address. Try again in {exc.retry_after} s."
        return error_response(request, 429, "rate_limited", message, exc.retry_after)

    @api.exception_handler(RateLimitedError)
    async def model_rate_limited(request: Request, exc: RateLimitedError) -> JSONResponse:
        retry_after = round(exc.retry_after) if exc.retry_after else MODEL_RETRY_SECONDS
        return error_response(request, 503, "model_rate_limited", str(exc), max(1, retry_after))

    @api.exception_handler(LLMError)
    async def model_error(request: Request, exc: LLMError) -> JSONResponse:
        """The client learns which kind of failure it was; the exact text goes to the log."""
        # An LLMError never holds a key, but it can name a setting or quote the model's reply.
        logger.warning(
            "model error",
            extra={"request_id": request.state.request_id, "detail": str(exc)[:500]},
        )
        if isinstance(exc, LLMConfigError):
            return error_response(
                request, 500, "model_not_configured", "The model is not set up on the server."
            )
        if isinstance(exc, StructuredOutputError):
            message = "The model's reply could not be used. Try again."
            return error_response(request, 502, "model_error", message)
        return error_response(request, 502, "model_error", str(exc))

    def limit_requests(request: Request) -> None:
        client = request.client.host if request.client else "unknown"
        retry_after = limiter.check(client)
        if retry_after is not None:
            raise TooManyRequests(retry_after)

    @api.get("/health")
    def health() -> dict[str, str]:
        """Liveness check."""
        return {"status": "ok"}

    @api.post("/ask", responses=ERRORS, dependencies=[Depends(limit_requests)])
    def ask(body: AskRequest, request: Request) -> AskResponse:
        """Answer a question. A plain `def`, so FastAPI runs the agent in its threadpool."""
        if not slots.acquire(blocking=False):
            raise Busy
        try:
            result = service.ask(body.question, request.state.request_id)
        finally:
            slots.release()
        request.state.log_fields = {
            "attempts": result.attempts,
            "refused": result.refused,
            "faithfulness_ok": result.faithfulness_ok,
            "rows": len(result.rows),
        }
        return result

    @api.get("/schema", responses={500: {"model": ErrorResponse}})
    def schema() -> SchemaResponse:
        """The tables and columns the agent can query, with their descriptions."""
        return service.schema()

    return api


app = create_app()
