TAX_RATE = 0.08

def compute_total(prices: list[float]) -> float:
    total = 0.0
    for price in prices:
        total += price * TAX_RATE
    return total
