import hashlib


def q586(candidate, verifier):
    """Wrap a controlled/mock verifier callback into structured outcomes. True means valid,
    False means counterexample; TimeoutError/SyntaxError/other exceptions map to
    timeout/compile_error/internal_error. Return status, candidate SHA256, and a bounded
    diagnostic. No untrusted code is executed by this adapter.
    """
    digest = hashlib.sha256(candidate.encode()).hexdigest()
    detail = ""
    try:
        result = verifier(candidate)
        if not isinstance(result, bool):
            raise TypeError("verifier must return bool")
        status = "valid" if result else "counterexample"
    except TimeoutError as error:
        status = "timeout"
        detail = str(error)
    except SyntaxError as error:
        status = "compile_error"
        detail = str(error)
    except Exception as error:
        status = "internal_error"
        detail = str(error)
    return {
        "status": status,
        "candidate_hash": digest,
        "detail": detail[:200],
    }
