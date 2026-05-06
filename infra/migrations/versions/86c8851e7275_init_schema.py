"""init schema

Revision ID: 86c8851e7275
Revises: 
Create Date: 2026-05-05 00:18:07.719657

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '86c8851e7275'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE repos (
            id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            source_url      TEXT,
            name            TEXT NOT NULL,
            default_branch  TEXT,
            commit_sha      TEXT,
            status          TEXT NOT NULL DEFAULT 'pending'
                              CHECK (status IN ('pending','indexing','ready','failed')),
            ingested_at     TIMESTAMPTZ NOT NULL DEFAULT now()
        );
    """)

    op.execute("""
        CREATE TABLE files (
            id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            repo_id     UUID NOT NULL REFERENCES repos(id) ON DELETE CASCADE,
            path        TEXT NOT NULL,
            language    TEXT NOT NULL,
            loc         INT  NOT NULL,
            UNIQUE (repo_id, path)
        );
        CREATE INDEX files_repo_id_idx ON files(repo_id);
    """)

    op.execute("""
        CREATE TABLE chunks (
            id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            file_id       UUID NOT NULL REFERENCES files(id) ON DELETE CASCADE,
            repo_id       UUID NOT NULL REFERENCES repos(id) ON DELETE CASCADE,
            symbol_name   TEXT,
            symbol_kind   TEXT
                            CHECK (symbol_kind IN ('function','class','method','window')),
            start_line    INT NOT NULL,
            end_line      INT NOT NULL,
            imports       TEXT[],
            content       TEXT NOT NULL,
            content_tsv   TSVECTOR
                            GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
            embedding     VECTOR(1536)
        );
        CREATE INDEX chunks_repo_id_idx     ON chunks(repo_id);
        CREATE INDEX chunks_file_id_idx     ON chunks(file_id);
        CREATE INDEX chunks_content_tsv_idx ON chunks USING gin (content_tsv);
        CREATE INDEX chunks_embedding_idx   ON chunks
                       USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
    """)

    op.execute("""
        CREATE TABLE queries (
            id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            repo_id     UUID NOT NULL REFERENCES repos(id) ON DELETE CASCADE,
            mode        TEXT NOT NULL
                          CHECK (mode IN ('explain','trace','debug','refactor')),
            question    TEXT NOT NULL,
            retrieved   JSONB NOT NULL,
            answer      TEXT NOT NULL,
            latency_ms  INT,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        CREATE INDEX queries_repo_id_idx ON queries(repo_id);
    """)

    op.execute("""
        CREATE TABLE judgments (
            id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            query_id     UUID NOT NULL REFERENCES queries(id) ON DELETE CASCADE,
            judge_model  TEXT NOT NULL,
            correctness  INT NOT NULL CHECK (correctness  BETWEEN 1 AND 5),
            relevance    INT NOT NULL CHECK (relevance    BETWEEN 1 AND 5),
            completeness INT NOT NULL CHECK (completeness BETWEEN 1 AND 5),
            rationale    TEXT,
            created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        CREATE INDEX judgments_query_id_idx ON judgments(query_id);
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS judgments;")
    op.execute("DROP TABLE IF EXISTS queries;")
    op.execute("DROP TABLE IF EXISTS chunks;")
    op.execute("DROP TABLE IF EXISTS files;")
    op.execute("DROP TABLE IF EXISTS repos;")
