def describe_user(user: dict | None) -> str:
    if user is None:
        return "anonymous"
    return user.get("name", "unknown")
