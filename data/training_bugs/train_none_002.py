def get_label(item: dict) -> str:
    if item.get("label") == None:
        return "unlabeled"
    return item["label"]
