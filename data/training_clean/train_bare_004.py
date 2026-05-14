def convert_float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return 0.0
