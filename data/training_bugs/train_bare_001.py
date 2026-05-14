def safe_parse_int(text: str) -> int | None:
    try:
        return int(text)
    except:
        return None
