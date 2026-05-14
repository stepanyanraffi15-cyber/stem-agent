def register_tag(tag: str, tags: list[str] | None = None) -> list[str]:
    if tags is None:
        tags = []
    if tag not in tags:
        tags.append(tag)
    return tags
