def merge_options(key: str, value: str, opts: dict[str, str] = {}) -> dict[str, str]:
    opts[key] = value
    return opts
