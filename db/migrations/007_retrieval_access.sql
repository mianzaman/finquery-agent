-- Retrieval tables: let the agent read them, and remember what each embedding was made from.

-- SHA-256 of the text that was embedded, so a rebuild only re-embeds rows whose text changed.
ALTER TABLE schema_docs ADD COLUMN content_hash TEXT;
ALTER TABLE query_examples ADD COLUMN content_hash TEXT;

-- The agent searches these as the read-only role. They hold descriptions and example queries,
-- nothing private; the role still cannot write to them.
GRANT SELECT ON schema_docs, query_examples TO agent_reader;
