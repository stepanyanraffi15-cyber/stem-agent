def swap(a: int, b: int) -> tuple[int, int]:
    temp = a
    a = b
    b = temp
    return a, a
