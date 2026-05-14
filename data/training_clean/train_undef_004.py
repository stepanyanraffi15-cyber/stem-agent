DISCOUNT_PCT = 0.10

def apply_discount(price: float) -> float:
    discounted = price - (price * DISCOUNT_PCT)
    return round(discounted, 2)
