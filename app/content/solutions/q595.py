def q595(records, prices):
    """Compute exact billing from category->integer token-count records and
    category->integer microcurrency-per-million-token prices. Sum integer numerators
    then round HALF UP once to microcurrency units. Return
    (exact_numerator,settled_units). Python integers avoid overflow; prices/counts must
    be nonnegative integers. Rounding each row instead changes totals.
    """
    numerator = 0
    for row in records:
        for category, tokens in row.items():
            price = prices[category]
            if (
                not isinstance(tokens, int)
                or not isinstance(price, int)
                or min(tokens, price) < 0
            ):
                raise ValueError("invalid integer billing")
            numerator += tokens * price
    # Round once after aggregation; rounding each category would change the total.
    return (numerator, (numerator + 500000) // 1000000)
