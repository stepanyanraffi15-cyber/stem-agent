def flatten(nested: list[list[int]]) -> list[int]:
    result: list[int] = []
    for sublist in nested:
        result.extend(sublist)
    return result
