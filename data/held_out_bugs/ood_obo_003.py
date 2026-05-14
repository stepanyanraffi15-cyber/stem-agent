def count_pairs(n: int) -> int:
    count = 0
    for i in range(n - 1):
        for j in range(i + 1, n - 1):
            count += 1
    return count
