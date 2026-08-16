CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id text NOT NULL,
    document_id text NOT NULL,
    chunking text NOT NULL,
    content text NOT NULL,
    token_start integer NOT NULL CHECK (token_start >= 0),
    token_end integer NOT NULL CHECK (token_end > token_start),
    source_uri text NOT NULL,
    heading text,
    module text,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    embedding_local vector(384),
    embedding_openai vector(1536),
    search_vector tsvector GENERATED ALWAYS AS
      (setweight(to_tsvector('english', coalesce(heading, '')), 'A') ||
       setweight(to_tsvector('english', content), 'B')) STORED,
    PRIMARY KEY (chunking, id)
) PARTITION BY LIST (chunking);

CREATE TABLE IF NOT EXISTS chunks_fixed_128_o16 PARTITION OF chunks FOR VALUES IN ('fixed_128_o16');
CREATE TABLE IF NOT EXISTS chunks_fixed_256_o32 PARTITION OF chunks FOR VALUES IN ('fixed_256_o32');
CREATE TABLE IF NOT EXISTS chunks_fixed_256_o96 PARTITION OF chunks FOR VALUES IN ('fixed_256_o96');
CREATE TABLE IF NOT EXISTS chunks_fixed_384_o64 PARTITION OF chunks FOR VALUES IN ('fixed_384_o64');
CREATE TABLE IF NOT EXISTS chunks_structural_256_o32 PARTITION OF chunks FOR VALUES IN ('structural_256_o32');

CREATE INDEX IF NOT EXISTS chunks_fts_idx ON chunks USING gin (search_vector);
CREATE INDEX IF NOT EXISTS chunks_module_idx ON chunks (module);

CREATE INDEX IF NOT EXISTS c128_local_hnsw ON chunks_fixed_128_o16 USING hnsw (embedding_local vector_cosine_ops) WITH (m=16, ef_construction=64);
CREATE INDEX IF NOT EXISTS c256l_local_hnsw ON chunks_fixed_256_o32 USING hnsw (embedding_local vector_cosine_ops) WITH (m=16, ef_construction=64);
CREATE INDEX IF NOT EXISTS c256h_local_hnsw ON chunks_fixed_256_o96 USING hnsw (embedding_local vector_cosine_ops) WITH (m=16, ef_construction=64);
CREATE INDEX IF NOT EXISTS c384_local_hnsw ON chunks_fixed_384_o64 USING hnsw (embedding_local vector_cosine_ops) WITH (m=16, ef_construction=64);
CREATE INDEX IF NOT EXISTS cstruct_local_hnsw ON chunks_structural_256_o32 USING hnsw (embedding_local vector_cosine_ops) WITH (m=16, ef_construction=64);

CREATE INDEX IF NOT EXISTS c128_openai_hnsw ON chunks_fixed_128_o16 USING hnsw (embedding_openai vector_cosine_ops) WITH (m=16, ef_construction=64);
CREATE INDEX IF NOT EXISTS c256l_openai_hnsw ON chunks_fixed_256_o32 USING hnsw (embedding_openai vector_cosine_ops) WITH (m=16, ef_construction=64);
CREATE INDEX IF NOT EXISTS c256h_openai_hnsw ON chunks_fixed_256_o96 USING hnsw (embedding_openai vector_cosine_ops) WITH (m=16, ef_construction=64);
CREATE INDEX IF NOT EXISTS c384_openai_hnsw ON chunks_fixed_384_o64 USING hnsw (embedding_openai vector_cosine_ops) WITH (m=16, ef_construction=64);
CREATE INDEX IF NOT EXISTS cstruct_openai_hnsw ON chunks_structural_256_o32 USING hnsw (embedding_openai vector_cosine_ops) WITH (m=16, ef_construction=64);

CREATE TABLE IF NOT EXISTS jobs (
    id uuid PRIMARY KEY,
    kind text NOT NULL CHECK (kind IN ('index_build', 'evaluation')),
    status text NOT NULL CHECK (status IN ('queued', 'running', 'succeeded', 'failed')),
    payload jsonb NOT NULL,
    progress double precision NOT NULL DEFAULT 0 CHECK (progress BETWEEN 0 AND 1),
    result jsonb,
    error jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    started_at timestamptz,
    finished_at timestamptz
);

CREATE INDEX IF NOT EXISTS jobs_claim_idx ON jobs (status, created_at) WHERE status = 'queued';

