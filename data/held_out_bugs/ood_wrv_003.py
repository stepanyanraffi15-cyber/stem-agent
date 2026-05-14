def split_evens_odds(nums: list[int]) -> tuple[list[int], list[int]]:
    evens: list[int] = []
    odds: list[int] = []
    for n in nums:
        if n % 2 == 0:
            evens.append(n)
        else:
            odds.append(n)
    return evens, evens
