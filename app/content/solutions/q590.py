def q590(threshold=3, cooldown=10, base_delay=1, max_delay=30):
    """Return a retry/circuit-breaker state machine with injected numeric time. allow(now)
    permits closed requests or ONE half-open probe; record(success,now) closes or
    counts/reopens failures. retry_delay(attempt,rng) returns full-jitter capped
    exponential delay. Caller enforces its attempt/deadline budget and classifies
    retryable errors; no sleeping occurs.
    """
    if threshold < 1 or min(cooldown, base_delay, max_delay) < 0:
        raise ValueError("invalid settings")

    class Breaker:

        def __init__(self):
            self.failures = 0
            self.state = "closed"
            self.opened = 0

        def allow(self, now):
            if self.state == "closed":
                return True
            if self.state == "open" and now - self.opened >= cooldown:
                self.state = "half_open"
                return True
            return False

        def record(self, success, now):
            if success:
                self.failures = 0
                self.state = "closed"
            else:
                self.failures += 1
                if self.state == "half_open" or self.failures >= threshold:
                    self.state = "open"
                    self.opened = now

        def retry_delay(self, attempt, rng):
            if attempt < 0:
                raise ValueError("negative attempt")
            return float(
                rng.uniform(0, min(max_delay, base_delay * 2 ** min(attempt, 60)))
            )

    return Breaker()
