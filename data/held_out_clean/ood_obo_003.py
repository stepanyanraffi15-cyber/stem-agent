def count_pairs(n: int) -> int:
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            count += 1
    return count
