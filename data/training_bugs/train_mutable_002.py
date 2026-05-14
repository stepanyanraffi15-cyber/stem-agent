def register_tag(tag: str, tags: list[str] = []) -> list[str]:
    if tag not in tags:
        tags.append(tag)
    return tags
