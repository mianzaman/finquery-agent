-- Retrieval tables. 384 dimensions fit both the local model and OpenAI text-embedding-3-small
-- called with dimensions=384, so switching the embedding provider needs no migration.

CREATE TABLE schema_docs (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    table_name      TEXT NOT NULL,
    column_name     TEXT,  -- NULL for a description of the table itself
    description     TEXT NOT NULL,
    embedding       vector(384),
    embedding_model TEXT,
    UNIQUE NULLS NOT DISTINCT (table_name, column_name)
);

CREATE TABLE query_examples (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    question        TEXT NOT NULL UNIQUE,
    sql             TEXT NOT NULL,
    embedding       vector(384),
    embedding_model TEXT
);

CREATE INDEX schema_docs_embedding_idx ON schema_docs USING hnsw (embedding vector_cosine_ops);
CREATE INDEX query_examples_embedding_idx ON query_examples USING hnsw (embedding vector_cosine_ops);
