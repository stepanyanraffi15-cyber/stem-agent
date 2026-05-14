def compute_total(prices: list[float]) -> float:
    total = 0.0
    for price in prices:
        total += price * tax_rate
    return total
