def apply_discount(price: float) -> float:
    discounted = price - (price * discount_pct)
    return round(discounted, 2)
