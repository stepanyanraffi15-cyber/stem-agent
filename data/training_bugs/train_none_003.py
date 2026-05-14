def describe_user(user: dict | None) -> str:
    if user == None:
        return "anonymous"
    return user.get("name", "unknown")
