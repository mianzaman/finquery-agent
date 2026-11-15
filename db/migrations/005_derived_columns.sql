-- Which values in a row were calculated by the loader rather than filed by the company,
-- as column names (for example {gross_profit}). Empty when every value is as filed.
-- free_cash_flow is always operating_cash_flow - capex and is not listed.

ALTER TABLE income_statements ADD COLUMN derived_columns TEXT[] NOT NULL DEFAULT '{}';
ALTER TABLE balance_sheets ADD COLUMN derived_columns TEXT[] NOT NULL DEFAULT '{}';
ALTER TABLE cash_flows ADD COLUMN derived_columns TEXT[] NOT NULL DEFAULT '{}';
