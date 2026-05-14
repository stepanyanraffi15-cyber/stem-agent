def safe_upper(text: str | None) -> str:
    if text is None:
        return ""
    return text.upper()
