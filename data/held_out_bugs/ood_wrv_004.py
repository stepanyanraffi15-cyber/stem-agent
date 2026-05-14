def compute_stats(data: list[float]) -> tuple[float, float]:
    mean = sum(data) / len(data)
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    std = variance ** 0.5
    return mean, mean
