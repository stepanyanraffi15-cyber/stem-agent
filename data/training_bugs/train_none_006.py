def first_non_null(items: list[object]) -> object:
    for item in items:
        if item != None:
            return item
    return None
