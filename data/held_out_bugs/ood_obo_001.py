def find_last(items: list[int], target: int) -> int:
    for i in range(len(items) - 1):
        if items[i] == target:
            return i
    return -1
