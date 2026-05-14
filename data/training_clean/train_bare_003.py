def fetch_value(data: dict, key: str) -> object:
    try:
        return data[key]
    except KeyError:
        return None
