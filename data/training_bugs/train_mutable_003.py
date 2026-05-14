def push_event(event: dict, log: list[dict] = []) -> list[dict]:
    log.append(event)
    return log
