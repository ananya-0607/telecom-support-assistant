def reciprocal_rank_fusion(rankings, constant=60):
    scores = {}
    for ranking in rankings:
        for rank, passage_id in enumerate(dict.fromkeys(ranking), 1):
            scores[passage_id] = scores.get(passage_id, 0) + 1/(constant+rank)
    return sorted(scores, key=lambda key: (-scores[key], key)), scores
