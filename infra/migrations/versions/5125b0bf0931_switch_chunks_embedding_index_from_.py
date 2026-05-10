"""switch chunks embedding index from ivfflat to hnsw

Revision ID: 5125b0bf0931
Revises: 86c8851e7275
Create Date: 2026-05-06 04:26:00.000000

Why: IVFFlat is an approximate index that requires its centroids to be
trained on representative data. We created the index in 001_init when
the table was empty, so the centroids were meaningless and queries with
selective WHERE filters returned 0-3 rows when more were expected.

HNSW (Hierarchical Navigable Small World) doesn't have this cold-start
problem -- it incrementally maintains its graph as rows are inserted,
so it works correctly regardless of when the index was built relative
to the data. It's also more accurate at the same speed.
"""
from alembic import op


revision = "5125b0bf0931"
down_revision = "86c8851e7275"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("DROP INDEX IF EXISTS chunks_embedding_idx;")
    op.execute("""
        CREATE INDEX chunks_embedding_idx
        ON chunks
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
    """)
    dbname = op.get_bind().exec_driver_sql("SELECT current_database()").scalar()
    op.execute(f"ALTER DATABASE {dbname} SET hnsw.ef_search = 100;")


def downgrade() -> None:
    dbname = op.get_bind().exec_driver_sql("SELECT current_database()").scalar()
    op.execute(f"ALTER DATABASE {dbname} RESET hnsw.ef_search;")
    op.execute("DROP INDEX IF EXISTS chunks_embedding_idx;")
    op.execute("""
        CREATE INDEX chunks_embedding_idx
        ON chunks
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);
    """)
