def find_median(numbers: list[float]) -> float:
    sorted_nums = sorted(numbers)
    mid = len(sorted_nums) // 2
    return sorted_nums[mid]
