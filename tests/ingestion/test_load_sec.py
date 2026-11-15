"""Tests for the SEC loader. Nothing here touches the network."""

import json
import os
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest
from sqlalchemy import text

from ingestion.load_sec import (
    MAX_ATTEMPTS,
    Period,
    SecClient,
    SecError,
    TagFacts,
    bases_differ,
    build_calendar,
    build_company,
    cik_by_ticker,
    derive_columns,
    fetch_json,
    fiscal_year_labels,
    index_tag,
    is_fresh,
    load,
    merge_related,
    quarter_values,
    read_companies,
    read_mapping,
    read_related,
    resolve_cik,
    store_company,
)

FIXTURE = Path(__file__).parent.parent / "fixtures" / "companyfacts_sample.json"
TICKERS_JSON = {
    "0": {"cik_str": 1001, "ticker": "ZZALFA", "title": "Alfa Corp"},
    "1": {"cik_str": 1002, "ticker": "ZZBETA", "title": "Beta Inc"},
    "2": {"cik_str": 1067983, "ticker": "BRK-B", "title": "Berkshire Hathaway"},
}


@pytest.fixture(scope="module")
def sample() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text(), parse_float=Decimal, parse_int=Decimal)


@pytest.fixture(scope="module")
def mapping() -> dict[str, dict[str, list[str]]]:
    return read_mapping()


def by_quarter(rows: list[dict[str, Any]], column: str) -> dict[tuple[int, int], Any]:
    return {(r["fiscal_year"], r["fiscal_quarter"]): r[column] for r in rows}


# --- Mapping ----------------------------------------------------------------------------------


def test_mapping_covers_every_financial_column(mapping: dict[str, dict[str, list[str]]]) -> None:
    assert set(mapping["income_statements"]) == {
        "revenue",
        "gross_profit",
        "operating_income",
        "net_income",
        "eps_diluted",
    }
    assert set(mapping["balance_sheets"]) == {
        "total_assets",
        "total_liabilities",
        "shareholders_equity",
        "cash",
        "long_term_debt",
    }
    # free_cash_flow is derived, not mapped.
    assert set(mapping["cash_flows"]) == {"operating_cash_flow", "capex"}
    tables = [mapping[name] for name in ("income_statements", "balance_sheets", "cash_flows")]
    assert all(tags for columns in tables for tags in columns.values())
    assert set(mapping["derived_inputs"]) == {
        "cost_of_revenue",
        "liabilities_and_equity",
        "total_equity",
        "capex_oil_and_gas",
        "capex_other_property",
    }


# --- Fiscal calendar --------------------------------------------------------------------------


def test_calendar_for_calendar_year_company(sample: dict[str, Any]) -> None:
    periods = build_calendar(sample["ALFA"], ["Revenues", "Assets"])

    assert [(p.fiscal_year, p.fiscal_quarter, p.end) for p in periods] == [
        (2023, 4, date(2023, 12, 31)),
        (2024, 1, date(2024, 3, 31)),
        (2024, 2, date(2024, 6, 30)),
        (2024, 3, date(2024, 9, 30)),
        (2024, 4, date(2024, 12, 31)),
    ]
    assert [p.start for p in periods[1:]] == [
        date(2024, 1, 1),
        date(2024, 4, 1),
        date(2024, 7, 1),
        date(2024, 10, 1),
    ]
    # Each quarter carries the day its own 10-Q or 10-K was filed, not a later comparative's.
    assert [p.filed for p in periods] == [
        date(2024, 2, 15),
        date(2024, 5, 1),
        date(2024, 8, 1),
        date(2024, 11, 1),
        date(2025, 2, 15),
    ]


def test_filing_date_is_stored_with_the_income_statement(
    sample: dict[str, Any], mapping: dict[str, dict[str, list[str]]]
) -> None:
    rows = build_company("ZZALFA", sample["ALFA"], mapping).rows

    assert by_quarter(rows["income_statements"], "filed_date")[(2024, 4)] == date(2025, 2, 15)
    assert "filed_date" not in rows["balance_sheets"][0]


def test_a_filing_without_a_date_leaves_the_filing_date_empty() -> None:
    facts = gaap(
        Revenues=[
            fact("2023-12-31", 400, "K23", "10-K", fy=2023, start="2023-01-01"),
            fact("2024-03-31", 100, "Q1", "10-Q", fy=2024, start="2024-01-01"),
        ]
    )
    for entry in facts["facts"]["us-gaap"]["Revenues"]["units"]["USD"]:
        entry.pop("filed", None)

    assert [p.filed for p in build_calendar(facts, ["Revenues"])] == [None, None]


def test_calendar_keeps_fiscal_year_separate_from_calendar_year(sample: dict[str, Any]) -> None:
    tags = ["RevenueFromContractWithCustomerExcludingAssessedTax"]
    periods = {(p.fiscal_year, p.fiscal_quarter): p for p in build_calendar(sample["BETA"], tags)}

    # Fiscal 2024 starts in calendar 2023.
    assert periods[(2024, 1)].end == date(2023, 12, 30)
    assert periods[(2024, 1)].start == date(2023, 10, 1)
    assert periods[(2024, 4)].end == date(2024, 9, 28)
    # A quarter filed after the last 10-K belongs to the next fiscal year.
    assert periods[(2025, 1)].end == date(2024, 12, 28)
    assert periods[(2025, 1)].start == date(2024, 9, 29)


def test_calendar_needs_an_annual_report() -> None:
    with pytest.raises(SecError, match="no 10-K"):
        build_calendar({"facts": {"us-gaap": {}}}, ["Revenues"])


# --- Quarter derivation -----------------------------------------------------------------------


def test_year_to_date_becomes_standalone_quarters(
    sample: dict[str, Any], mapping: dict[str, dict[str, list[str]]]
) -> None:
    data = build_company("ALFA", sample["ALFA"], mapping)

    # Net income and cash flow are filed year to date only: 10, 22, 36 and 52 for the year.
    assert by_quarter(data.rows["income_statements"], "net_income") == {
        (2024, 1): 10,
        (2024, 2): 12,
        (2024, 3): 14,
        (2024, 4): 16,
    }
    assert by_quarter(data.rows["cash_flows"], "operating_cash_flow") == {
        (2024, 1): 20,
        (2024, 2): 25,
        (2024, 3): 30,
        (2024, 4): 35,
    }


def test_three_month_fact_is_preferred_and_q4_comes_from_the_10k(
    sample: dict[str, Any], mapping: dict[str, dict[str, list[str]]]
) -> None:
    data = build_company("ALFA", sample["ALFA"], mapping)

    assert by_quarter(data.rows["income_statements"], "revenue") == {
        (2024, 1): 100,  # as first filed; a later 10-Q/A said 102
        (2024, 2): 110,  # filed as a three-month figure
        (2024, 3): 120,  # nine months 330 - six months 210
        (2024, 4): 150,  # full year 480 - nine months 330
    }
    assert by_quarter(data.rows["income_statements"], "eps_diluted")[(2024, 4)] == Decimal("1.6")


def test_q4_when_only_three_month_facts_were_filed(
    sample: dict[str, Any], mapping: dict[str, dict[str, list[str]]]
) -> None:
    data = build_company("BETA", sample["BETA"], mapping)

    # Quarters 200, 210, 220; full year 900.
    assert by_quarter(data.rows["income_statements"], "revenue") == {
        (2024, 1): 200,
        (2024, 2): 210,
        (2024, 3): 220,
        (2024, 4): 270,
        (2025, 1): 230,
    }


def test_no_quarter_is_invented_without_the_earlier_quarters(
    sample: dict[str, Any], mapping: dict[str, dict[str, list[str]]]
) -> None:
    data = build_company("ALFA", sample["ALFA"], mapping)

    # Fiscal 2023 has a full-year figure but no 10-Qs, so there is no Q4 to derive.
    assert (2023, 4) not in by_quarter(data.rows["income_statements"], "revenue")
    assert by_quarter(data.rows["balance_sheets"], "total_assets")[(2023, 4)] == 990


# --- Tag fallback, balance sheet, derived columns ---------------------------------------------


def test_tag_fallback_is_per_quarter(
    sample: dict[str, Any], mapping: dict[str, dict[str, list[str]]]
) -> None:
    alfa = build_company("ALFA", sample["ALFA"], mapping)
    beta = build_company("BETA", sample["BETA"], mapping)

    cash = by_quarter(alfa.rows["balance_sheets"], "cash")
    assert cash[(2024, 1)] == 50  # first tag wins over `Cash` (49)
    assert cash[(2024, 2)] == 55  # only `Cash` has this quarter
    assert cash[(2024, 3)] is None
    # BETA never uses `Revenues`; the second tag supplies revenue.
    assert by_quarter(beta.rows["income_statements"], "revenue")[(2024, 1)] == 200
    assert by_quarter(beta.rows["balance_sheets"], "cash")[(2024, 4)] == 300


def test_balance_sheet_takes_the_value_at_period_end(
    sample: dict[str, Any], mapping: dict[str, dict[str, list[str]]]
) -> None:
    data = build_company("ALFA", sample["ALFA"], mapping)

    assert by_quarter(data.rows["balance_sheets"], "total_assets") == {
        (2023, 4): 990,
        (2024, 1): 1000,
        (2024, 2): 1010,
        (2024, 3): 1020,
        (2024, 4): 1030,
    }
    assert "period_start" not in data.rows["balance_sheets"][0]


def test_free_cash_flow_and_period_columns(
    sample: dict[str, Any], mapping: dict[str, dict[str, list[str]]]
) -> None:
    alfa = build_company("ALFA", sample["ALFA"], mapping)
    beta = build_company("BETA", sample["BETA"], mapping)

    assert by_quarter(alfa.rows["cash_flows"], "free_cash_flow") == {
        (2024, 1): 15,
        (2024, 2): 19,
        (2024, 3): 23,
        (2024, 4): 27,
    }
    q4 = next(r for r in alfa.rows["income_statements"] if r["fiscal_quarter"] == 4)
    assert (q4["period_start"], q4["period_end"]) == (date(2024, 10, 1), date(2024, 12, 31))
    assert alfa.fiscal_year_end_month == 12
    assert beta.fiscal_year_end_month == 9
    assert beta.rows["cash_flows"] == []


def test_missing_columns_are_reported(
    sample: dict[str, Any], mapping: dict[str, dict[str, list[str]]]
) -> None:
    data = build_company("ALFA", sample["ALFA"], mapping)

    assert data.missing_columns == [
        "income_statements.gross_profit",
        "income_statements.operating_income",
        "balance_sheets.total_liabilities",
        "balance_sheets.long_term_debt",
    ]


def test_only_the_last_five_fiscal_years_are_kept(mapping: dict[str, dict[str, list[str]]]) -> None:
    facts = [
        {
            "end": f"{year}-12-31",
            "val": year,
            "accn": f"K{year}",
            "fy": year,
            "fp": "FY",
            "form": "10-K",
            "filed": f"{year + 1}-02-01",
        }
        for year in range(2015, 2025)
    ]
    company = {"facts": {"us-gaap": {"Assets": {"units": {"USD": facts}}}}}

    data = build_company("OLD", company, mapping)

    assert sorted(r["fiscal_year"] for r in data.rows["balance_sheets"]) == [
        2020,
        2021,
        2022,
        2023,
        2024,
    ]


def test_an_open_fiscal_year_does_not_use_up_the_five_year_window(
    mapping: dict[str, dict[str, list[str]]],
) -> None:
    def fact(year: int, end: str, form: str, accn: str) -> dict[str, Any]:
        return {"end": end, "val": 1, "accn": accn, "fy": year, "form": form, "filed": end}

    facts = [fact(year, f"{year}-12-31", "10-K", f"K{year}") for year in range(2015, 2025)]
    facts.append(fact(2025, "2025-03-31", "10-Q", "Q1-2025"))
    company = {"facts": {"us-gaap": {"Assets": {"units": {"USD": facts}}}}}

    data = build_company("OLD", company, mapping)

    assert sorted({r["fiscal_year"] for r in data.rows["balance_sheets"]}) == [
        2020,
        2021,
        2022,
        2023,
        2024,
        2025,
    ]


# --- Fiscal year labels, per-share values, derived columns, related registrants ---------------


def gaap(**tags: list[dict[str, Any]]) -> dict[str, Any]:
    """A company-facts file with the given `Tag=[facts]`; the unit follows the tag name."""
    body = {}
    for tag, rows in tags.items():
        unit = "USD/shares" if tag.startswith("EarningsPerShare") else "USD"
        body[tag] = {"units": {unit: rows}}
    return {"facts": {"us-gaap": body}}


def fact(
    end: str, val: Any, accn: str, form: str = "10-K", fy: int | None = None, start: str = ""
) -> dict[str, Any]:
    row = {"end": end, "val": val, "accn": accn, "form": form, "fy": fy, "filed": accn[-10:]}
    return {"start": start, **row} if start else row


def test_one_wrong_fiscal_year_label_is_outvoted() -> None:
    # SEC labels the 10-K for 2021 as fy=2020, as it does for Honeywell.
    sec_labels = {
        date(2019, 12, 31): 2019,
        date(2020, 12, 31): 2020,
        date(2021, 12, 31): 2020,
        date(2022, 12, 31): 2022,
        date(2023, 12, 31): None,
    }

    assert list(fiscal_year_labels(sec_labels).values()) == [2019, 2020, 2021, 2022, 2023]


def test_fiscal_year_labels_keep_the_company_convention_and_survive_gaps() -> None:
    # A retailer that calls the year ending January 2024 "fiscal 2023"; one 10-K is absent.
    sec_labels = {date(2021, 1, 31): 2020, date(2022, 1, 30): 2021, date(2024, 1, 28): 2023}

    assert list(fiscal_year_labels(sec_labels).values()) == [2020, 2021, 2023]


def test_every_value_is_kept_as_first_filed() -> None:
    company = gaap(
        EarningsPerShareDiluted=[
            fact("2025-06-30", "2.45", "Q2-2025-07-24", "10-Q", start="2025-04-01"),
            fact("2025-06-30", "4.9", "Q2-2026-07-23", "10-Q", start="2025-04-01"),  # after a split
        ],
        Revenues=[
            fact("2025-06-30", 100, "Q2-2025-07-24", "10-Q", start="2025-04-01"),
            fact("2025-06-30", 52, "Q2-2026-07-23", "10-Q", start="2025-04-01"),  # after a spin-off
        ],
        Assets=[
            fact("2025-06-30", 900, "Q2-2025-07-24", "10-Q"),
            fact("2025-06-30", 905, "K-2026-02-01"),
        ],
    )
    end, start = date(2025, 6, 30), date(2025, 4, 1)

    eps = index_tag(company, "EarningsPerShareDiluted")
    revenue = index_tag(company, "Revenues")

    assert eps.durations[end][start] == Decimal("2.45")
    assert revenue.durations[end][start] == 100
    assert revenue.source[(start, end)] == "Q2-2025-07-24"
    assert revenue.as_filed["Q2-2026-07-23"][(start, end)] == 52
    assert index_tag(company, "Assets").instants[end] == 900


def two_years(recast_nine_months: int) -> tuple[TagFacts, list[Period], list[Period]]:
    """Capex filed year to date. The 2026 Q3 filing repeats 2025's nine months as given."""

    def year(label: int) -> list[Period]:
        start = date(label, 1, 1)
        ends = [date(label, 3, 31), date(label, 6, 30), date(label, 9, 30)]
        return [Period(label, q + 1, None, end, start) for q, end in enumerate(ends)]

    def ytd(label: int, quarter: int, value: int, accn: str) -> dict[str, Any]:
        end = year(label)[quarter - 1].end.isoformat()
        return fact(end, value, accn, "10-Q", start=f"{label}-01-01")

    company = gaap(
        Capex=[
            ytd(2025, 1, 100, "Q1-2025-04-25"),
            ytd(2025, 2, 210, "Q2-2025-07-25"),
            ytd(2025, 3, 330, "Q3-2025-10-25"),
            ytd(2026, 1, 110, "Q1-2026-04-25"),
            ytd(2025, 1, 100, "Q1-2026-04-25"),
            ytd(2026, 2, 230, "Q2-2026-07-25"),
            ytd(2025, 2, 210, "Q2-2026-07-25"),
            ytd(2026, 3, 180, "Q3-2026-10-25"),
            ytd(2025, 3, recast_nine_months, "Q3-2026-10-25"),
        ]
    )
    return index_tag(company, "Capex"), year(2026), year(2025)


def test_quarters_are_derived_when_history_is_unchanged() -> None:
    tag, this_year, last_year = two_years(recast_nine_months=330)

    values, blocked = quarter_values(tag, this_year, last_year)

    assert values == {1: 110, 2: 120, 3: -50}
    assert blocked == {}


@pytest.mark.parametrize("recast_nine_months", [150, 325, 340])
def test_a_quarter_is_not_derived_across_a_restatement(recast_nine_months: int) -> None:
    # The Q3 filing restates last year's nine months (a spin-off), so this year's nine months
    # are on a new basis while the six-month figure it would be subtracted from is not.
    tag, this_year, last_year = two_years(recast_nine_months)

    values, blocked = quarter_values(tag, this_year, last_year)

    assert values == {1: 110, 2: 120}
    assert set(blocked) == {3}


def test_a_tiny_restatement_is_ignored() -> None:
    tag, this_year, last_year = two_years(recast_nine_months=331)

    assert quarter_values(tag, this_year, last_year)[1] == {}


def test_a_figure_filed_for_the_quarter_survives_a_restatement() -> None:
    tag, this_year, last_year = two_years(recast_nine_months=150)
    tag.durations[date(2026, 9, 30)][date(2026, 7, 1)] = Decimal(60)

    values, blocked = quarter_values(tag, this_year, last_year)

    assert values[3] == 60
    assert blocked == {}


def test_restated_quarters_become_warnings_and_are_not_filled_from_another_tag() -> None:
    def ytd(end: str, value: int, accn: str, form: str = "10-Q") -> dict[str, Any]:
        return fact(end, value, accn, form, fy=int(end[:4]), start=f"{end[:4]}-01-01")

    company = gaap(
        Assets=[
            fact("2024-12-31", 1, "K-2025-02-01", fy=2024),
            fact("2025-12-31", 1, "K-2026-02-01", fy=2025),
            fact("2025-09-30", 1, "Q3-2025-10-25", "10-Q"),
            fact("2026-06-30", 1, "Q2-2026-07-25", "10-Q"),
            fact("2026-09-30", 1, "Q3-2026-10-25", "10-Q"),
        ],
        PaymentsToAcquirePropertyPlantAndEquipment=[
            ytd("2025-09-30", 330, "Q3-2025-10-25"),
            ytd("2026-06-30", 230, "Q2-2026-07-25"),
            ytd("2026-09-30", 180, "Q3-2026-10-25"),
            ytd("2025-09-30", 150, "Q3-2026-10-25"),
        ],
        PaymentsToAcquireProductiveAssets=[
            fact("2026-09-30", 999, "Q3-2026-10-25", "10-Q", start="2026-07-01")
        ],
    )

    data = build_company("SPN", company, read_mapping())

    capex = by_quarter(data.rows["cash_flows"], "capex") if data.rows["cash_flows"] else {}
    assert (2026, 3) not in capex or capex[(2026, 3)] is None
    assert data.warnings == [
        "cash_flows.capex left empty for fiscal 2026 Q3: its filing restated last year's figure "
        "by -55% (spin-off, stock split or reclassification), so the quarter cannot be derived"
    ]


@pytest.mark.parametrize(
    ("now", "before", "differ"),
    [
        ("1", "1", False),  # nothing restated
        ("1.005", "1", False),  # rounding
        ("0.5", "1", True),  # spin-off or split first shown in this filing
        ("0.1", "0.1", False),  # split earlier in the year: both filings on the new basis
        ("0.49", "0.48", False),  # the quarter after a spin-off
        ("0.48", "0.90", True),  # a second spin-off
        ("1.3", "1", True),
    ],
)
def test_bases_differ(now: str, before: str, differ: bool) -> None:
    assert bases_differ(Decimal(now), Decimal(before)) is differ


def test_capex_is_summed_for_oil_and_gas_producers_only_when_both_lines_are_filed(
    mapping: dict[str, Any],
) -> None:
    span = {"start": "2024-10-01", "form": "10-Q"}
    both = gaap(
        PaymentsToAcquireOilAndGasPropertyAndEquipment=[
            fact("2024-12-31", 1400, "Q-2025-02-01", **span)
        ],
        PaymentsToAcquireOtherPropertyPlantAndEquipment=[
            fact("2024-12-31", 100, "Q-2025-02-01", **span)
        ],
    )
    one = gaap(
        PaymentsToAcquireOilAndGasPropertyAndEquipment=[
            fact("2024-12-31", 1400, "Q-2025-02-01", **span)
        ]
    )
    key = (2024, 4)
    summed: dict[str, Any] = {"cash_flows": {"capex": {}}}
    left_empty: dict[str, Any] = {"cash_flows": {"capex": {}}}

    marked = derive_columns(both, mapping["derived_inputs"], quarters_of(both), summed)
    unmarked = derive_columns(one, mapping["derived_inputs"], quarters_of(one), left_empty)

    assert summed["cash_flows"]["capex"] == {key: 1500}
    assert marked == {("cash_flows", "capex", key)}
    assert left_empty["cash_flows"]["capex"] == {}
    assert unmarked == set()


def derived_company() -> dict[str, Any]:
    year = {"start": "2024-01-01"}
    return gaap(
        Revenues=[fact("2024-12-31", 1000, "K-2025-02-01", fy=2024, **year)],
        CostOfRevenue=[fact("2024-12-31", 600, "K-2025-02-01", fy=2024, **year)],
        Assets=[fact("2024-12-31", 5000, "K-2025-02-01", fy=2024)],
        LiabilitiesAndStockholdersEquity=[fact("2024-12-31", 5000, "K-2025-02-01", fy=2024)],
        StockholdersEquity=[fact("2024-12-31", 1900, "K-2025-02-01", fy=2024)],
        StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest=[
            fact("2024-12-31", 2000, "K-2025-02-01", fy=2024)
        ],
    )


def quarters_of(company: dict[str, Any]) -> list[Period]:
    start = date(2024, 1, 1)
    return [Period(2024, 4, date(2024, 10, 1), date(2024, 12, 31), start)]


def test_total_liabilities_is_calculated_from_equity_including_minorities(
    mapping: dict[str, Any],
) -> None:
    data = build_company("DRV", derived_company(), mapping)

    row = data.rows["balance_sheets"][0]
    assert row["total_liabilities"] == 3000  # 5000 - 2000, not 5000 - 1900
    assert row["derived_columns"] == ["total_liabilities"]
    assert "balance_sheets.total_liabilities" not in data.missing_columns


def test_gross_profit_is_calculated_only_when_cost_of_revenue_is_filed(
    mapping: dict[str, Any],
) -> None:
    key = (2024, 4)
    with_cost: dict[str, Any] = {"income_statements": {"revenue": {key: 1000}, "gross_profit": {}}}
    bank: dict[str, Any] = {"income_statements": {"revenue": {key: 1000}, "gross_profit": {}}}
    company = gaap(
        CostOfRevenue=[fact("2024-12-31", 600, "Q-2025-02-01", "10-Q", start="2024-10-01")]
    )

    marked = derive_columns(company, mapping["derived_inputs"], quarters_of(company), with_cost)
    unmarked = derive_columns(gaap(), mapping["derived_inputs"], quarters_of(company), bank)

    assert with_cost["income_statements"]["gross_profit"] == {key: 400}
    assert marked == {("income_statements", "gross_profit", key)}
    assert bank["income_statements"]["gross_profit"] == {}
    assert unmarked == set()


def test_a_filed_value_is_never_replaced_by_a_calculated_one(mapping: dict[str, Any]) -> None:
    company = derived_company()
    company["facts"]["us-gaap"]["Liabilities"] = {
        "units": {"USD": [fact("2024-12-31", 2950, "K-2025-02-01", fy=2024)]}
    }

    row = build_company("DRV", company, mapping).rows["balance_sheets"][0]

    assert row["total_liabilities"] == 2950
    assert row["derived_columns"] == []


def parent_and_subsidiary() -> tuple[dict[str, Any], dict[str, Any]]:
    parent = gaap(
        Assets=[
            fact("2025-12-31", 900, "PARENT-K-2026-02-01", fy=2025),
            fact("2026-03-31", 910, "PARENT-Q1-2026-05-01", "10-Q"),
        ]
    )
    subsidiary = gaap(
        Assets=[
            # The subsidiary's own balance sheet, from its own filing.
            fact("2025-12-31", 300, "SUB-K-2026-02-01", fy=2025),
            fact("2026-06-30", 310, "SUB-Q2-2026-08-01", "10-Q"),
            # The combined 10-Q: the parent's numbers, filed under the subsidiary.
            fact("2025-12-31", 900, "COMBINED-Q2-2026-08-01", "10-Q"),
            fact("2026-06-30", 925, "COMBINED-Q2-2026-08-01", "10-Q"),
            # The next quarter repeats the same year-end figure.
            fact("2025-12-31", 900, "COMBINED-Q3-2026-11-01", "10-Q"),
            fact("2026-09-30", 940, "COMBINED-Q3-2026-11-01", "10-Q"),
            # Already in the parent's file: must not be added twice.
            fact("2026-03-31", 910, "PARENT-Q1-2026-05-01", "10-Q"),
        ],
        Revenues=[
            fact("2026-06-30", 77, "COMBINED-Q2-2026-08-01", "10-Q", start="2026-04-01"),
            fact("2026-06-30", 11, "SUB-Q2-2026-08-01", "10-Q", start="2026-04-01"),
        ],
    )
    return parent, subsidiary


def test_co_registrant_filings_are_taken_only_when_they_match_the_parent() -> None:
    parent, subsidiary = parent_and_subsidiary()

    added = merge_related(parent, subsidiary, ["Assets"], require_match=True)

    assert added == 2
    assets = index_tag(parent, "Assets").instants
    assert assets[date(2026, 6, 30)] == 925  # not the subsidiary's 310
    assert assets[date(2026, 9, 30)] == 940
    assert assets[date(2025, 12, 31)] == 900  # not the subsidiary's 300
    revenue = index_tag(parent, "Revenues").durations[date(2026, 6, 30)]
    assert list(revenue.values()) == [77]
    assert len(parent["facts"]["us-gaap"]["Assets"]["units"]["USD"]) == 6


def test_predecessor_filings_are_all_taken() -> None:
    successor = gaap(Assets=[fact("2026-06-30", 500, "NEW-Q2-2026-08-01", "10-Q")])
    predecessor = gaap(
        Assets=[
            fact("2019-12-31", 300, "OLD-K-2020-02-01", fy=2019),
            fact("2025-12-31", 480, "OLD-K-2026-02-01", fy=2025),
        ]
    )

    assert merge_related(successor, predecessor, ["Assets"], require_match=False) == 2
    assert set(index_tag(successor, "Assets").instants) == {
        date(2019, 12, 31),
        date(2025, 12, 31),
        date(2026, 6, 30),
    }


def test_related_ciks_file_is_well_formed() -> None:
    related = read_related()

    assert related["XOM"] == {"predecessors": [34088]}
    for ticker, kinds in related.items():
        assert ticker == ticker.upper()
        assert set(kinds) <= {"predecessors", "co_registrants"}
        assert all(isinstance(cik, int) for ciks in kinds.values() for cik in ciks)


# --- Inputs -----------------------------------------------------------------------------------


def test_read_companies(tmp_path: Path) -> None:
    path = tmp_path / "companies.csv"
    path.write_text("ticker,name\naapl,Apple Inc.\n MSFT ,Microsoft\n")

    assert read_companies(path) == [("AAPL", "Apple Inc."), ("MSFT", "Microsoft")]


def test_read_companies_empty_and_wrong_header(tmp_path: Path) -> None:
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    wrong = tmp_path / "wrong.csv"
    wrong.write_text("symbol,title\nAAPL,Apple\n")

    assert read_companies(empty) == []
    with pytest.raises(SecError, match="ticker,name"):
        read_companies(wrong)


@pytest.mark.parametrize(
    ("ticker", "cik"), [("ZZALFA", 1001), ("BRK.B", 1067983), ("BRK-B", 1067983)]
)
def test_resolve_cik(ticker: str, cik: int) -> None:
    assert resolve_cik(ticker, cik_by_ticker(TICKERS_JSON)) == cik


def test_resolve_cik_unknown_ticker() -> None:
    with pytest.raises(SecError, match="NOPE"):
        resolve_cik("NOPE", cik_by_ticker(TICKERS_JSON))


# --- HTTP and cache ---------------------------------------------------------------------------


def make_client(handler: Any, sleeps: list[float] | None = None) -> SecClient:
    record = sleeps if sleeps is not None else []
    return SecClient(
        "Test Suite test@example.com",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=record.append,
    )


def test_client_sends_user_agent() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["User-Agent"])
        return httpx.Response(200, text="{}")

    assert make_client(handler).get_text("https://data.sec.gov/x") == "{}"
    assert seen == ["Test Suite test@example.com"]


def test_client_requires_user_agent() -> None:
    with pytest.raises(SecError, match="SEC_USER_AGENT"):
        SecClient("  ")


@pytest.mark.parametrize("status", [429, 500, 503])
def test_client_retries_then_succeeds(status: int) -> None:
    statuses = [status, status, 200]
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(statuses.pop(0), text="ok")

    assert make_client(handler, sleeps).get_text("https://data.sec.gov/x") == "ok"
    assert statuses == []
    assert [s for s in sleeps if s >= 1] == [1.0, 2.0]  # exponential backoff


def test_client_gives_up_after_bounded_attempts() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(429)

    with pytest.raises(SecError, match="HTTP 429"):
        make_client(handler).get_text("https://data.sec.gov/x")
    assert len(calls) == MAX_ATTEMPTS


def test_client_does_not_retry_not_found() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(404)

    with pytest.raises(SecError, match="HTTP 404"):
        make_client(handler).get_text("https://data.sec.gov/x")
    assert len(calls) == 1


def test_client_throttles_to_five_requests_per_second() -> None:
    now = [0.0]
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    client = SecClient(
        "Test Suite test@example.com",
        client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200))),
        sleep=sleep,
        clock=lambda: now[0],
    )
    for _ in range(3):
        client.get_text("https://data.sec.gov/x")

    assert sleeps == [pytest.approx(0.2), pytest.approx(0.2)]


def test_cache_is_used_until_it_is_seven_days_old(tmp_path: Path) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, text='{"n": 1.5}')

    client = make_client(handler)
    path = tmp_path / "sec" / "X.json"

    assert fetch_json(client, "https://data.sec.gov/x", path, refresh=False) == {
        "n": Decimal("1.5")
    }
    fetch_json(client, "https://data.sec.gov/x", path, refresh=False)
    assert len(calls) == 1  # second read came from the cache

    fetch_json(client, "https://data.sec.gov/x", path, refresh=True)
    assert len(calls) == 2  # --refresh ignores the cache

    old = (datetime.now(tz=UTC) - timedelta(days=8)).timestamp()
    os.utime(path, (old, old))
    assert not is_fresh(path, datetime.now(tz=UTC))
    fetch_json(client, "https://data.sec.gov/x", path, refresh=False)
    assert len(calls) == 3  # stale cache is downloaded again


# --- Database ---------------------------------------------------------------------------------


@pytest.fixture
def clean_companies(db: None) -> Any:
    """Remove the fixture companies before and after a database test."""
    from app.db.session import admin_engine

    def wipe() -> None:
        with admin_engine.begin() as conn:
            for table in ("income_statements", "balance_sheets", "cash_flows", "companies"):
                conn.execute(text(f"DELETE FROM {table} WHERE ticker IN ('ZZALFA', 'ZZBETA')"))

    wipe()
    yield admin_engine
    wipe()


def test_upsert_is_idempotent_and_updates_changes(
    clean_companies: Any, sample: dict[str, Any], mapping: dict[str, dict[str, list[str]]]
) -> None:
    engine = clean_companies
    data = build_company("ZZALFA", sample["ALFA"], mapping)
    expected = 1 + sum(len(rows) for rows in data.rows.values())

    assert store_company(engine, "ZZALFA", "Alfa Corp", 1001, data) == expected
    assert store_company(engine, "ZZALFA", "Alfa Corp", 1001, data) == 0

    data.rows["income_statements"][0]["revenue"] += 1
    assert store_company(engine, "ZZALFA", "Alfa Corp", 1001, data) == 1

    with engine.connect() as conn:
        stored = conn.execute(
            text(
                "SELECT revenue, net_income, eps_diluted, period_start, period_end "
                "FROM income_statements "
                "WHERE ticker = 'ZZALFA' AND fiscal_year = 2024 AND fiscal_quarter = 4"
            )
        ).one()
        month = conn.execute(
            text("SELECT fiscal_year_end_month FROM companies WHERE ticker = 'ZZALFA'")
        ).scalar_one()
    assert tuple(stored) == (150, 16, Decimal("1.6"), date(2024, 10, 1), date(2024, 12, 31))
    assert month == 12


def test_load_runs_end_to_end_and_survives_a_bad_company(
    clean_companies: Any,
    sample: dict[str, Any],
    mapping: dict[str, dict[str, list[str]]],
    tmp_path: Path,
) -> None:
    # The fixture file holds two companies; serve each one from its own URL, as SEC would.
    raw = json.loads(FIXTURE.read_text())
    bodies = {
        "/api/xbrl/companyfacts/CIK0000001001.json": raw["ALFA"],
        "/api/xbrl/companyfacts/CIK0000001002.json": raw["BETA"],
    }

    def serve(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("company_tickers.json"):
            return httpx.Response(200, json=TICKERS_JSON)
        if request.url.path in bodies:
            return httpx.Response(200, json=bodies[request.url.path])
        return httpx.Response(404)

    companies = [("ZZALFA", "Alfa Corp"), ("NOPE", "Not Listed"), ("ZZBETA", "Beta Inc")]

    first = load(clean_companies, make_client(serve), companies, mapping, raw_dir=tmp_path)
    second = load(clean_companies, make_client(serve), companies, mapping, raw_dir=tmp_path)

    assert "NOPE" in first["NOPE"]["error"]
    assert first["ZZALFA"]["changed_rows"] > 0
    assert first["ZZBETA"]["rows"] == {"income_statements": 5, "balance_sheets": 6, "cash_flows": 0}
    assert "balance_sheets.total_liabilities" in first["ZZALFA"]["missing_columns"]
    assert second["ZZALFA"]["changed_rows"] == 0
    assert second["ZZBETA"]["changed_rows"] == 0
