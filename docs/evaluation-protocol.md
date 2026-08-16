# Frozen evaluation protocol

1. Ingest the manifest-pinned Python 3.14.6 documentation.
2. Author and independently inspect 300 queries; approve only correct questions and source
   spans. Split by stable ID into 100 development and 200 test queries, stratified by category
   and difficulty.
3. Tune chunking, RRF, HNSW, and reranking only on development data. Record every sweep in
   MLflow.
4. Freeze test JSONL and manifest. Select parameters before revealing final test results.
5. Run all nine configurations against the same corpus, qrels, hardware profile, top-k values,
   cache policy, and warm-up policy.
6. Compare global and sliced quality, p50/p95/p99, throughput, build time, index bytes, peak
   RSS, provider tokens/searches, and estimated USD using the committed pricing snapshot.
7. Inspect twenty errors spanning all categories and identify which method wins or fails.
8. Recommend one configuration each for maximum quality, minimum latency, and minimum cost.

The test set is not a tuning oracle. Any code path setting both `split: test` and
`tuning: true`, or observing a checksum mismatch, fails before retrieval begins.

