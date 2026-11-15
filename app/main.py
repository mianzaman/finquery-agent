"""FastAPI app and routes."""

from fastapi import FastAPI

app = FastAPI(title="FinQuery Agent")


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check."""
    return {"status": "ok"}
