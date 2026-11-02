import json


def q589(connection, key, arguments, delta):
    """Implement an idempotent MOCK effect where both the effect counter and dedup result
    live in the SAME SQLite transaction. Repeated key+canonical JSON arguments returns
    prior result; changed arguments with the same key raise. Return counter value after
    effect. This atomic local service does not prove exactly-once behavior for an
    unrelated remote API.
    """
    canonical = json.dumps(
        {"arguments": arguments, "delta": delta},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    with connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS effects (key TEXT PRIMARY KEY,args TEXT,result INTEGER)"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS counter (id INTEGER PRIMARY KEY,value INTEGER)"
        )
        connection.execute("INSERT OR IGNORE INTO counter VALUES (1,0)")
    with connection:
        # Dedup lookup and the local effect share one write transaction.
        connection.execute("BEGIN IMMEDIATE")
        old = connection.execute(
            "SELECT args,result FROM effects WHERE key=?", (key,)
        ).fetchone()
        if old:
            if old[0] != canonical:
                raise ValueError("idempotency key reused with changed arguments")
            return old[1]
        connection.execute("UPDATE counter SET value=value+? WHERE id=1", (delta,))
        result = connection.execute("SELECT value FROM counter WHERE id=1").fetchone()[
            0
        ]
        connection.execute(
            "INSERT INTO effects VALUES (?,?,?)", (key, canonical, result)
        )
    return result
