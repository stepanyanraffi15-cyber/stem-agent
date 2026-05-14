SCALE_FACTOR = 2.0

def scale_vector(values: list[float]) -> list[float]:
    return [v * SCALE_FACTOR for v in values]
