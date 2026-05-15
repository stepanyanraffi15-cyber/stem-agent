def load_config(path: str) -> dict:
    try:
        with open(path) as f:
            import json
            return json.load(f)
    except:
        return {}
