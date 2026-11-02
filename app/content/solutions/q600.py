def q600():
    """Return a small integer-price/quantity reference limit-order book with price-time
    priority, cancel, and GTC/IOC/FOK. submit(id,side,price,quantity,tif) returns trades
    (resting_id,price,quantity) and unfilled quantity; FOK checks liquidity before
    mutation. IDs are never reused. O(N log N) sorting is intentional; this does not
    claim the Rust engine's complexity.
    """

    class Book:

        def __init__(self):
            self.orders = {}
            self.seen = set()
            self.clock = 0

        def cancel(self, key):
            return self.orders.pop(key, None)

        def submit(self, key, side, price, quantity, tif="GTC"):
            if (
                key in self.seen
                or side not in ("buy", "sell")
                or tif not in ("GTC", "IOC", "FOK")
                or (not isinstance(price, int))
                or (not isinstance(quantity, int))
                or (price < 0)
                or (quantity <= 0)
            ):
                raise ValueError("invalid order")
            candidates = [
                (order_id, resting_order)
                for order_id, resting_order in self.orders.items()
                if resting_order["side"] != side
                and (
                    resting_order["price"] <= price
                    if side == "buy"
                    else resting_order["price"] >= price
                )
            ]
            candidates.sort(
                key=lambda pair: (
                    (pair[1]["price"] if side == "buy" else -pair[1]["price"]),
                    pair[1]["time"],
                )
            )
            self.seen.add(key)
            if (
                tif == "FOK"
                and sum((resting_order["quantity"] for _, resting_order in candidates))
                < quantity
            ):
                return ([], quantity)
            # FOK liquidity is checked before any resting order quantity is changed.
            trades = []
            remaining = quantity
            for order_id, order in candidates:
                amount = min(remaining, order["quantity"])
                remaining -= amount
                order["quantity"] -= amount
                trades.append((order_id, order["price"], amount))
                if order["quantity"] == 0:
                    del self.orders[order_id]
                if remaining == 0:
                    break
            if remaining and tif == "GTC":
                self.orders[key] = {
                    "side": side,
                    "price": price,
                    "quantity": remaining,
                    "time": self.clock,
                }
                self.clock += 1
            return (trades, remaining)

    return Book()
