def moving_average(data: list[float], window: int) -> list[float]:
    result = []
    for i in range(len(data) - window + 1):
        window_sum = sum(data[i:i + window])
        result.append(window_sum / window)
    return result
