import hashlib
import json


def q591(proposal, approval):
    """Validate an immutable proposal approval. Proposal has id,version,arguments; approval
    has id,version,argument_hash,decision. Canonicalize arguments to SHA256, and return
    whether approved ID/version/hash all match plus current hash. Persistence and effect
    dispatch are external; argument changes invalidate approval.
    """
    payload = json.dumps(
        proposal["arguments"],
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    # Canonical serialization makes approvals sensitive to argument changes, not key order.
    digest = hashlib.sha256(payload.encode()).hexdigest()
    allowed = (
        approval is not None
        and approval.get("decision") == "approved"
        and (approval.get("id") == proposal["id"])
        and (approval.get("version") == proposal["version"])
        and (approval.get("argument_hash") == digest)
    )
    return (allowed, digest)
