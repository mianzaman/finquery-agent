-- Stock splits, and earnings per share put on today's share basis.

-- ratio is new shares per old share: 10 for a 10-for-1 split, 0.125 for a 1-for-8 reverse split.
CREATE TABLE stock_splits (
    ticker     TEXT NOT NULL REFERENCES companies (ticker),
    split_date DATE NOT NULL,
    ratio      NUMERIC(12, 6) NOT NULL CHECK (ratio > 0),
    PRIMARY KEY (ticker, split_date)
);

-- The day the 10-Q or 10-K that first reported the quarter was filed. A report filed after a
-- split already states per-share figures on the new basis, even for a quarter that ended
-- before the split, so the filing date (not period_end) decides which splits still apply.
ALTER TABLE income_statements ADD COLUMN filed_date DATE;

-- income_statements with eps_diluted_adjusted: eps_diluted divided by every split that took
-- effect after the quarter was filed (after period_end when the filing date is unknown).
-- eps_diluted itself stays as reported. split_factor is the divisor, 1 when no split applies.
CREATE VIEW income_statements_split_adjusted AS
SELECT
    i.ticker,
    i.fiscal_year,
    i.fiscal_quarter,
    i.period_start,
    i.period_end,
    i.filed_date,
    i.revenue,
    i.gross_profit,
    i.operating_income,
    i.net_income,
    i.eps_diluted,
    (i.eps_diluted / f.split_factor)::NUMERIC(20, 4) AS eps_diluted_adjusted,
    f.split_factor,
    i.derived_columns
FROM income_statements AS i
CROSS JOIN LATERAL (
    -- Postgres has no product aggregate; exp(sum(ln(x))) on NUMERIC is exact to 6 places here.
    SELECT COALESCE(ROUND(EXP(SUM(LN(s.ratio))), 6), 1) AS split_factor
    FROM stock_splits AS s
    WHERE s.ticker = i.ticker
      AND s.split_date > COALESCE(i.filed_date, i.period_end)
) AS f;

GRANT SELECT ON stock_splits, income_statements_split_adjusted TO agent_reader;
