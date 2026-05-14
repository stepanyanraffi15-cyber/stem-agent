def push_event(event: dict, log: list[dict] | None = None) -> list[dict]:
    if log is None:
        log = []
    log.append(event)
    return log
