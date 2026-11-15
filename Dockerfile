FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.8 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

WORKDIR /srv

# Dependencies first, so this layer is reused until the lockfile changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY app ./app
COPY db ./db
COPY ingestion ./ingestion
COPY evals ./evals
COPY ui ./ui

# COPY keeps the host's file modes, and some source files are private there (0600): make the
# code readable by the unprivileged user. /srv/data holds the reply cache (a volume) and the
# embedding model (mounted read-only).
RUN useradd --create-home --uid 10001 appuser \
    && chmod -R a+rX /srv/app /srv/db /srv/ingestion /srv/evals /srv/ui \
    && mkdir -p /srv/data \
    && chown appuser:appuser /srv/data
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
