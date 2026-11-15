-- Data tables. Money is USD in NUMERIC(20,4); fiscal and calendar periods are separate columns.

CREATE TABLE companies (
    ticker                TEXT PRIMARY KEY,
    cik                   BIGINT UNIQUE,
    name                  TEXT NOT NULL,
    sector                TEXT,
    industry              TEXT,
    fiscal_year_end_month SMALLINT CHECK (fiscal_year_end_month BETWEEN 1 AND 12)
);

CREATE TABLE income_statements (
    ticker           TEXT NOT NULL REFERENCES companies (ticker),
    fiscal_year      SMALLINT NOT NULL,
    fiscal_quarter   SMALLINT NOT NULL CHECK (fiscal_quarter BETWEEN 1 AND 4),
    period_start     DATE,
    period_end       DATE NOT NULL,
    revenue          NUMERIC(20, 4),
    gross_profit     NUMERIC(20, 4),
    operating_income NUMERIC(20, 4),
    net_income       NUMERIC(20, 4),
    eps_diluted      NUMERIC(20, 4),
    PRIMARY KEY (ticker, fiscal_year, fiscal_quarter)
);

CREATE TABLE balance_sheets (
    ticker              TEXT NOT NULL REFERENCES companies (ticker),
    fiscal_year         SMALLINT NOT NULL,
    fiscal_quarter      SMALLINT NOT NULL CHECK (fiscal_quarter BETWEEN 1 AND 4),
    period_end          DATE NOT NULL,
    total_assets        NUMERIC(20, 4),
    total_liabilities   NUMERIC(20, 4),
    shareholders_equity NUMERIC(20, 4),
    cash                NUMERIC(20, 4),
    long_term_debt      NUMERIC(20, 4),
    PRIMARY KEY (ticker, fiscal_year, fiscal_quarter)
);

CREATE TABLE cash_flows (
    ticker              TEXT NOT NULL REFERENCES companies (ticker),
    fiscal_year         SMALLINT NOT NULL,
    fiscal_quarter      SMALLINT NOT NULL CHECK (fiscal_quarter BETWEEN 1 AND 4),
    period_end          DATE NOT NULL,
    operating_cash_flow NUMERIC(20, 4),
    capex               NUMERIC(20, 4),
    free_cash_flow      NUMERIC(20, 4),
    PRIMARY KEY (ticker, fiscal_year, fiscal_quarter)
);

CREATE TABLE daily_prices (
    ticker     TEXT NOT NULL REFERENCES companies (ticker),
    trade_date DATE NOT NULL,
    open       NUMERIC(20, 4),
    high       NUMERIC(20, 4),
    low        NUMERIC(20, 4),
    close      NUMERIC(20, 4),
    adj_close  NUMERIC(20, 4),
    volume     BIGINT,
    PRIMARY KEY (ticker, trade_date)
);

CREATE TABLE dividends (
    ticker  TEXT NOT NULL REFERENCES companies (ticker),
    ex_date DATE NOT NULL,
    amount  NUMERIC(20, 4) NOT NULL,
    PRIMARY KEY (ticker, ex_date)
);

CREATE INDEX income_statements_ticker_period_end_idx ON income_statements (ticker, period_end);
CREATE INDEX balance_sheets_ticker_period_end_idx ON balance_sheets (ticker, period_end);
CREATE INDEX cash_flows_ticker_period_end_idx ON cash_flows (ticker, period_end);
CREATE INDEX daily_prices_trade_date_idx ON daily_prices (trade_date);
