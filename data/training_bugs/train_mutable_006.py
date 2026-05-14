def track_visit(page: str, history: list[str] = []) -> list[str]:
    history.append(page)
    return history
