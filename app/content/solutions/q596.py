import threading


def q596(limit):
    """Return a thread-safe single-tenant integer budget ledger with reserve/settle/refund.
    Reservations use unique request IDs; replay must match amounts. Reject over-budget
    reservations. Settlement records actual incurred cost even above estimate/limit,
    leaving negative availability if necessary; it cannot erase provider usage. This is
    in-memory, not durable accounting.
    """
    if not isinstance(limit, int) or limit < 0:
        raise ValueError("nonnegative integer limit")

    class Ledger:

        def __init__(self):
            self.rows = {}
            self.lock = threading.RLock()

        def available(self):
            with self.lock:
                return limit - sum(
                    (row[1] for row in self.rows.values() if row[0] != "refunded")
                )

        def reserve(self, key, amount):
            if not isinstance(amount, int) or amount < 0:
                raise ValueError("invalid amount")
            with self.lock:
                if key in self.rows:
                    if self.rows[key] != ("reserved", amount):
                        raise ValueError("conflicting replay")
                    return
                if amount > self.available():
                    raise ValueError("insufficient budget")
                self.rows[key] = ("reserved", amount)

        def settle(self, key, actual):
            if not isinstance(actual, int) or actual < 0:
                raise ValueError("invalid actual")
            with self.lock:
                state, amount = self.rows[key]
                if state == "settled" and amount == actual:
                    return
                if state != "reserved":
                    raise ValueError("invalid settlement transition")
                self.rows[key] = ("settled", actual)

        def refund(self, key):
            with self.lock:
                state, amount = self.rows[key]
                if state == "refunded":
                    return
                if state != "reserved":
                    raise ValueError("only unspent reservation can be refunded")
                self.rows[key] = ("refunded", 0)

    return Ledger()
