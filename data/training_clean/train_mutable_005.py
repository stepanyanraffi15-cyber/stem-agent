def merge_options(key: str, value: str, opts: dict[str, str] | None = None) -> dict[str, str]:
    if opts is None:
        opts = {}
    opts[key] = value
    return opts
