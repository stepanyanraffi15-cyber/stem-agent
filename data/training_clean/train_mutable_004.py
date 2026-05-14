def add_score(score: int, scores: list[int] | None = None) -> list[int]:
    if scores is None:
        scores = []
    scores.append(score)
    return sorted(scores)
