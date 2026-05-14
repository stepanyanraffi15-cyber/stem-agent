def normalize(values: list[float]) -> list[float]:
    lo = min(values)
    hi = max(values)
    return [(v - lo) / (hi - lo) for v in values]
