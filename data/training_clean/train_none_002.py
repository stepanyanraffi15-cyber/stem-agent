def get_label(item: dict) -> str:
    if item.get("label") is None:
        return "unlabeled"
    return item["label"]
