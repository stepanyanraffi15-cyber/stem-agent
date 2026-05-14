def track_visit(page: str, history: list[str] | None = None) -> list[str]:
    if history is None:
        history = []
    history.append(page)
    return history
