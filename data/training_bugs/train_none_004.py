def validate_token(token: str | None) -> bool:
    if token != None and len(token) > 0:
        return True
    return False
