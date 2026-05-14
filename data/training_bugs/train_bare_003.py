def fetch_value(data: dict, key: str) -> object:
    try:
        return data[key]
    except:
        return None
