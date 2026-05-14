def add_score(score: int, scores: list[int] = []) -> list[int]:
    scores.append(score)
    return sorted(scores)
