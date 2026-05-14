def first_non_null(items: list[object]) -> object:
    for item in items:
        if item is not None:
            return item
    return None
