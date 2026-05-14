def safe_upper(text: str | None) -> str:
    if text == None:
        return ""
    return text.upper()
