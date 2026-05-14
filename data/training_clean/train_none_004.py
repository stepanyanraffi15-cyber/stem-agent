def validate_token(token: str | None) -> bool:
    if token is not None and len(token) > 0:
        return True
    return False
