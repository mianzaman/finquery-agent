-- Read-only role for the agent. It is created without a login or password here;
-- db/migrate.py enables login and sets the password from READER_DATABASE_URL at run time.

DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'agent_reader') THEN
        CREATE ROLE agent_reader NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT;
    END IF;
END
$$;

-- PUBLIC can create temporary tables by default; nobody but the owner needs that here.
DO $$
BEGIN
    EXECUTE format('REVOKE TEMPORARY ON DATABASE %I FROM PUBLIC', current_database());
    EXECUTE format('GRANT CONNECT ON DATABASE %I TO agent_reader', current_database());
END
$$;

REVOKE CREATE ON SCHEMA public FROM PUBLIC;
GRANT USAGE ON SCHEMA public TO agent_reader;

GRANT SELECT ON
    companies,
    income_statements,
    balance_sheets,
    cash_flows,
    daily_prices,
    dividends
TO agent_reader;

-- Defence in depth: the engine also sets this on connect.
ALTER ROLE agent_reader SET statement_timeout = '5s';
