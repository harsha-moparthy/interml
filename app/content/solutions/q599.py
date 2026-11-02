import hashlib
import json


def q599(tenant, model_revision, messages, tools, parameters):
    """Build an isolated inference-cache key from tenant/access scope, explicit model
    revision, ordered messages/tools, and all effective sampling parameters.
    Canonicalize dictionary keys but preserve list order; reject NaN JSON values. Return
    SHA256. TTL storage and stochastic-request caching policy are outside this
    key-construction core.
    """
    payload = {
        "tenant": tenant,
        "model": model_revision,
        "messages": messages,
        "tools": tools,
        "parameters": parameters,
    }
    # Sort object keys but preserve array order; both affect cache identity.
    text = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
