def q598(providers, request):
    """Fail over between supplied provider(request)->chunk-iterator callbacks. A
    RuntimeError before any NONEMPTY emitted chunk permits trying the next provider;
    after any output return a terminal partial-stream error, never concatenate another
    provider's answer. Return collected chunks, attempt log, and status. This
    deterministic core treats RuntimeError as retryable; provider selection/billing are
    external.
    """
    attempts = []
    for name, provider in providers:
        output = []
        try:
            for chunk in provider(request):
                if chunk:
                    output.append(chunk)
            attempts.append((name, "success"))
            return (output, attempts, "success")
        except RuntimeError:
            attempts.append(
                (
                    name,
                    "partial_failure" if output else "pre_output_failure",
                )
            )
            # Once output is visible, switching providers would splice incompatible answers.
            if output:
                return (output, attempts, "partial_failure")
    return ([], attempts, "unavailable")
