# Evaluation dataset

The benchmark requires 300 human-verified English queries: 100 development and 200 test.
The repository intentionally does not label generated examples as human-verified. Prepare
`all.jsonl`, review every draft with `retrieval-benchmark dataset review`, then run
`retrieval-benchmark dataset freeze`. Commit the resulting `dev.jsonl`, `test.jsonl`, and
`test.manifest.json`; do not commit corpus archives, parsed documents, or embeddings.

Each JSONL record is validated by `BenchmarkQuery` and contains stable ID, question,
category, difficulty, split, graded relevant source spans, optional chunk projections,
provenance, generation method, reviewer, validation timestamp, and status.
Source `start`/`end` spans use the parser's deterministic whitespace-token offsets.

The test set is immutable after freezing. Its manifest stores the canonical file checksum,
and test-mode evaluation refuses a mismatching file.
