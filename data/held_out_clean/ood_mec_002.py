def safe_head(items: list[object]) -> object | None:
    if not items:
        return None
    return items[0]
