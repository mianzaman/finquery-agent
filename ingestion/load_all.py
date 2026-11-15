"""Runs every loader in order: SEC financials, then prices, dividends and sectors.

Run with `uv run python -m ingestion.load_all [--tickers AAPL,MSFT] [--refresh]`. The prices step
reads its tickers from the `companies` table, so it is skipped when the SEC step fails.
"""

import argparse
import logging
import sys

from ingestion import load_prices, load_sec

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Load SEC financials, then prices.")
    parser.add_argument("--tickers", help="comma-separated subset of ingestion/companies.csv")
    parser.add_argument("--refresh", action="store_true", help="download again, ignoring the cache")
    args = parser.parse_args(argv)
    forwarded = (["--tickers", args.tickers] if args.tickers else []) + (
        ["--refresh"] if args.refresh else []
    )

    code = load_sec.main(forwarded)
    if code != 0:
        logger.error("SEC load failed; prices not loaded")
        return code
    return load_prices.main(forwarded)


if __name__ == "__main__":
    sys.exit(main())
