# Engineering Notes

Selected debugging stories and design decisions from building CodeCopilot.
These are short, honest writeups of the bugs and tradeoffs that took the
longest to resolve — the ones that taught me the most.

---

## 1. IVFFlat cold-start: when an index lies about empty results

I started with `pgvector`'s IVFFlat index because every tutorial uses it.
After ingesting a repo, queries with selective `WHERE` filters returned
*fewer rows than they should*. Same query without the filter worked fine.

Three wrong hypotheses before I went to `EXPLAIN ANALYZE`:
1. The filter syntax was wrong (no — it ran)
2. The embeddings were corrupted (no — direct comparison worked)
3. Connection pool reuse was caching stale plans (no — fresh connections same result)

`EXPLAIN ANALYZE` showed the planner was returning ~10 candidates from
the IVFFlat index, then filtering down to a handful. Root cause: IVFFlat
trains its centroid clusters at index creation time, on whatever data
exists at that moment. I'd created the index *before* ingesting, so its
centroids were meaningless. Probes hit clusters that didn't actually
contain the closest matches.

**Fix:** migrated to HNSW via Alembic. HNSW builds its graph incrementally
on inserts, no cold-start problem. Wrote it up as a versioned migration
so the fix is portable.

**Takeaway:** Approximate indexes have implicit assumptions about the
data distribution at creation time. Vector indexes especially. When
results don't match expectations, run `EXPLAIN ANALYZE` before assuming
your code is wrong.

---

## 2. Calibrated uncertainty: when the right answer is "no"

Every retrieval-augmented generation system has a failure mode where it
fabricates plausible-sounding answers from irrelevant chunks. To prevent
this, I added explicit refusal logic in two places:

1. **In the reranker prompt:** the model assigns 1–5 relevance scores
   with one-sentence justifications. Scores aren't normalized to a
   distribution — they're absolute.
2. **In the answer prompt:** if no chunk reaches relevance 4 or higher,
   the answer mode steers the model to refuse and explain why, rather
   than answering tangentially.

This was tested by asking my system about JWT authentication in a
codebase that has *no JWT code*. The system correctly responded:
"The chunks don't show JWT-specific handling. What this codebase
provides is the OAuth2 bearer-token infrastructure that typically
wraps JWT usage..."

That's the calibrated answer. Low confidence is signaled honestly,
and the user is told what *is* available so they can refine the
question.

**Takeaway:** RAG systems that always answer are worse than RAG
systems that sometimes refuse. The refusal mode is a feature, not
a fallback.

---

## 3. The eval system is the project's most valuable artifact

After Phase 2, the system worked — but I had no way to measure
whether *my changes were making it better*. So I built a
three-dimensional LLM-as-judge:

- **Correctness:** are the claims supported by the retrieved chunks?
- **Relevance:** does the answer address the actual question?
- **Completeness:** is critical information from the chunks present?

Each dimension is scored 1–5 with a one-sentence rationale. The
judge sees the question, the chunks, and the answer; nothing else.

I hand-labeled a 20-question "golden set" spanning all four answer
modes and ran it as a baseline. The result — **4.33/5 overall** —
isn't the impressive part. The impressive part is that the score is
*reproducible*. I can change one prompt, re-run the script, get a
new number, compare directly. That comparability is what evaluation
means.

**Takeaway:** A frozen golden set is more valuable than the model
weights. Code and models can be rebuilt; the test set encodes what
"good" looks like for the product.

---

## 4. pgbouncer transaction-pooling is incompatible with prepared statements

When deploying to a managed Postgres (Supabase) that fronts the DB
with `pgbouncer` in transaction-pooling mode, my asyncpg-based code
hung on the first DB write after each new connection.

The error trail led me through three layered fixes:

1. **`statement_cache_size=0` on the asyncpg pool.** Disables
   client-side caching of prepared statements. asyncpg uses prepared
   statements for performance; the pooler can't preserve them across
   transaction boundaries, so they fail.
2. **Replaced `executemany` with `copy_records_to_table`.**
   `executemany` *also* uses prepared statements internally, so it has
   the same problem. The COPY protocol bypasses prepared statements
   entirely. (Later reverted when I switched to the session pooler,
   which doesn't have this constraint.)
3. **Switched from transaction pooler (port 6543) to session pooler
   (port 5432).** Session-mode pgbouncer holds a connection for the
   client's whole session, behaving like a regular Postgres connection.
   Standard asyncpg patterns work without changes.

**Takeaway:** "Postgres" through a pooler is *not the same as* direct
Postgres. The connection-handling abstraction leaks in subtle ways.
When something works locally and breaks in production, the connection
layer is one of the first places to look.

---

## 5. Schema-portable migrations

My initial Alembic migration had a hardcoded database name in
`ALTER DATABASE codecopilot SET hnsw.ef_search = 100`. This worked
locally where the database is literally called `codecopilot`. On
Supabase, the database is `postgres`, so the migration crashed.

The fix is one line:

    dbname = op.get_bind().exec_driver_sql("SELECT current_database()").scalar()
    op.execute(f"ALTER DATABASE {dbname} SET hnsw.ef_search = 100;")

**Takeaway:** Migrations should never hardcode environment-specific
identifiers — database names, schema names, role names. Fetch them
from the catalog at migration time. The same migration should run
unchanged in dev, staging, and production, regardless of how those
environments name their databases.

---

## 6. The eval system told me where to focus next

After running the baseline, I sorted the results by lowest score.
Two questions tied for last (avg 2.67 and 3.0):

- "Why might a path operation return 422 unexpectedly?"
- "Suggest a refactor for repeated parameter analysis logic."

In both cases, *correctness* was 2 but *relevance* was 4–5. The
model was trying to answer the right question, but the chunks it
had to work with didn't actually contain the right code. I dug into
the retrieval data and found that for the 422 question, the retriever
hadn't surfaced `exception_handlers.py` — exactly the file where
`RequestValidationError` lives.

This is a retrieval problem, not a generation problem. No amount of
prompt engineering on the answer model would fix it; the answer can't
cite chunks that weren't retrieved. The fix has to be in the
retrieval pipeline — better keyword expansion, alternative embedding
strategies for question-style queries, or a query rewriter.

**Takeaway:** Without per-dimension scoring, this would have looked
like "the answer mode is bad" and I might have wasted hours on the
prompts. The decomposed rubric pointed directly at retrieval as the
real bottleneck. Decomposed metrics save days.
