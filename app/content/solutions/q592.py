def q592(crash_at="after_effect", idempotent=True):
    """Simulate crash/recovery at before_effect, after_effect, or after_checkpoint for ONE
    logical operation. Persistent state is modeled by dictionaries outside the restarted
    worker. Return effect_count and checkpoint_done after recovery. An idempotent mock
    service prevents duplicate effects; a non-idempotent service demonstrates the
    after-effect gap. This is a deterministic boundary simulation, not process-crash
    integration testing.
    """
    if crash_at not in (
        "before_effect",
        "after_effect",
        "after_checkpoint",
    ):
        raise ValueError("unknown boundary")
    state = {"effect_count": 0, "done": False, "keys": set()}

    def worker(crash=None):
        if state["done"]:
            return
        if crash == "before_effect":
            return
        if not idempotent or "step1" not in state["keys"]:
            state["effect_count"] += 1
            state["keys"].add("step1")
        if crash == "after_effect":
            return
        state["done"] = True

    worker(crash_at)
    worker()
    return (state["effect_count"], state["done"])
