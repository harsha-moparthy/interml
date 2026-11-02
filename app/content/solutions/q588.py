import json
import sqlite3


def q588(path, dependencies):
    """Return a SQLite-backed workflow-state store. Dependencies map node to prerequisite
    IDs; require an acyclic graph. claim() transactionally claims one ready pending
    node; finish(node,success) persists terminal state; recover() explicitly resets
    interrupted running nodes to pending. Schema/dependency configuration must match on
    reopen. This core assumes recovery is called only when old workers are stopped;
    effects are a separate exercise.
    """
    names = set(dependencies)
    pending = set(names)
    done = set()
    while pending:
        ready = {
            node_name for node_name in pending if set(dependencies[node_name]) <= done
        }
        if not ready:
            raise ValueError("cycle or missing dependency")
        done |= ready
        pending -= ready
    config = json.dumps({k: sorted(v) for k, v in dependencies.items()}, sort_keys=True)

    class Store:

        def __init__(self):
            self.db = sqlite3.connect(path)
            with self.db:
                self.db.execute(
                    "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)"
                )
                self.db.execute(
                    "CREATE TABLE IF NOT EXISTS nodes (id TEXT PRIMARY KEY,state TEXT,attempts INTEGER)"
                )
                old = self.db.execute(
                    "SELECT value FROM metadata WHERE key=?", ("schema1",)
                ).fetchone()
                if old and old[0] != config:
                    raise ValueError("workflow configuration mismatch")
                self.db.execute(
                    "INSERT OR IGNORE INTO metadata VALUES (?,?)",
                    ("schema1", config),
                )
                self.db.executemany(
                    "INSERT OR IGNORE INTO nodes VALUES (?, ?, ?)",
                    [(node_name, "pending", 0) for node_name in names],
                )

        def states(self):
            return {
                node_name: (node_status, attempt_count)
                for node_name, node_status, attempt_count in self.db.execute(
                    "SELECT id,state,attempts FROM nodes"
                )
            }

        def claim(self):
            with self.db:
                self.db.execute("BEGIN IMMEDIATE")
                state = self.states()
                for node_name in sorted(names):
                    if state[node_name][0] == "pending" and all(
                        (
                            state[dependency][0] == "succeeded"
                            for dependency in dependencies[node_name]
                        )
                    ):
                        self.db.execute(
                            "UPDATE nodes SET state='running',attempts=attempts+1 WHERE id=?",
                            (node_name,),
                        )
                        return node_name
            return None

        def finish(self, node, success):
            with self.db:
                cursor = self.db.execute(
                    "UPDATE nodes SET state=? WHERE id=? AND state='running'",
                    ("succeeded" if success else "failed", node),
                )
                if cursor.rowcount != 1:
                    raise ValueError("node not running")

        def recover(self):
            with self.db:
                self.db.execute(
                    "UPDATE nodes SET state='pending' WHERE state='running'"
                )

        def close(self):
            self.db.close()

    return Store()
