def min_max(values: list[float]) -> tuple[float, float]:
    lo = values[0]
    hi = values[0]
    for v in values[1:]:
        if v < lo:
            lo = v
        if v > hi:
            hi = v
    return lo, hi
