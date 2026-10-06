from ingest import bm25, collection, experiments, tokenise

def hybrid_retrieve_experiments(query: str, top_k: int = 3,
                                 exclude_id: str = None, k: int = 60) -> list:
    """RRF fusion of BM25 + vector, with optional self-exclusion filter."""

    # ── BM25 ranking ──────────────────────────────────────────────────
    tokens = tokenise(query)
    bm25_scores = bm25.get_scores(tokens)
    bm25_ranked = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)
    bm25_ranked = [
        (i, score) for i, score in bm25_ranked
        if experiments[i]["id"] != exclude_id
    ]

    # ── Vector (ChromaDB) ranking ────────────────────────────────────
    n_results = len(experiments) - (1 if exclude_id else 0)
    where_filter = {"id": {"$ne": exclude_id}} if exclude_id else None

    vector_results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where_filter
    )
    vector_ranked_ids = vector_results["ids"][0]

    # ── RRF Fusion ────────────────────────────────────────────────────
    rrf_scores: dict[str, float] = {}

    for rank, (i, _) in enumerate(bm25_ranked, start=1):
        exp_id = experiments[i]["id"]
        rrf_scores[exp_id] = rrf_scores.get(exp_id, 0.0) + 1 / (k + rank)

    for rank, exp_id in enumerate(vector_ranked_ids, start=1):
        rrf_scores[exp_id] = rrf_scores.get(exp_id, 0.0) + 1 / (k + rank)

    sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    exp_by_id = {e["id"]: e for e in experiments}
    return [
        {"experiment": exp_by_id[exp_id], "rrf_score": round(score, 6)}
        for exp_id, score in sorted_ids[:top_k]
    ]

# # Smoke test
# # q = "XGBoost binary classification with potential target leakage in features"
# q = "credit scoring model with demographic disparity fairness concern"
# results = hybrid_retrieve_experiments(q)
# print(f"\nQuery: {q}")
# for r in results:
#     e = r["experiment"]
#     print(f"  [{r['rrf_score']}] {e['id']} — {e['model']} ({e['outcome']}) leakage={e['leakage_risk']}")

# # 🎯 HINT: Try querying for bias-related experiments:
# # hybrid_retrieve_experiments("credit scoring model with demographic disparity fairness concern")
# # Do the bias-rejected experiments surface at the top?
