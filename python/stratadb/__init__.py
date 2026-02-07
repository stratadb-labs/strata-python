"""
StrataDB Python SDK

An embedded database for AI agents with six primitives: KV Store, Event Log,
State Cell, JSON Store, Vector Store, and Branches.

Example usage:

    from stratadb import Strata

    db = Strata.open("/data")
    db.kv_put("user:123", "Alice")

    # Branch isolation
    db.create_branch("experiment")
    db.set_branch("experiment")

    # Vector search with NumPy
    import numpy as np
    embedding = np.random.rand(384).astype(np.float32)
    db.vector_create_collection("docs", 384)
    db.vector_upsert("docs", "doc-1", embedding)
    results = db.vector_search("docs", embedding, k=5)

    # Transactions
    with db.transaction():
        db.kv_put("a", 1)
        db.kv_put("b", 2)
    # Auto-commits on success, auto-rollbacks on exception

Limitations:
    - **Bytes roundtrip via bundles:** ``Value.Bytes`` (Python ``bytes``) stored in
      the database will survive normal get/put operations, but a bundle
      export → import cycle serializes data through JSON. Because JSON has no
      native binary type, byte values are base64-encoded on export and decoded
      as ``str`` on import. After a roundtrip the value will come back as a
      Python ``str`` instead of ``bytes``.
"""

from ._stratadb import Strata as _Strata

# Exception hierarchy
from ._stratadb import (
    StrataError,
    NotFoundError,
    ValidationError,
    ConflictError,
    StateError,
    ConstraintError,
    AccessDeniedError,
    IoError,
)


class Transaction:
    """Context manager for database transactions.

    Automatically commits on success and rolls back on exception.

    Usage:
        with db.transaction() as txn:
            db.kv_put("key1", "value1")
            db.kv_put("key2", "value2")
        # Auto-commit on exit

        with db.transaction(read_only=True):
            value = db.kv_get("key1")
        # Read-only transaction
    """

    def __init__(self, db: "_Strata", read_only: bool = False):
        self._db = db
        self._read_only = read_only

    def __enter__(self) -> "Transaction":
        self._db.begin(self._read_only)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is None:
            # Success - commit
            self._db.commit()
        else:
            # Exception - rollback
            self._db.rollback()
        return False  # Don't suppress exceptions


def setup() -> str:
    """Download model files for auto-embedding.

    Downloads MiniLM-L6-v2 model files (~80MB) to ~/.stratadb/models/minilm-l6-v2/.
    Called automatically when ``auto_embed=True`` is passed to ``Strata.open()``,
    but can be called explicitly to pre-download during installation.

    Returns:
        The path where model files are stored.

    Raises:
        RuntimeError: If the download fails.

    Example::

        import stratadb
        stratadb.setup()  # pre-download model files
    """
    return _Strata.setup()


class Strata(_Strata):
    """StrataDB database handle with transaction support.

    This extends the native Strata class with a Pythonic transaction()
    context manager.
    """

    def transaction(self, read_only: bool = False) -> Transaction:
        """Start a transaction as a context manager.

        Usage:
            with db.transaction() as txn:
                db.kv_put("key1", "value1")
                db.kv_put("key2", "value2")
            # Auto-commit on exit

            with db.transaction(read_only=True):
                value = db.kv_get("key1")
            # Read-only transaction

        Args:
            read_only: If True, creates a read-only transaction.

        Returns:
            A Transaction context manager.

        Raises:
            StateError: If a transaction is already active.
        """
        return Transaction(self, read_only)


__all__ = [
    "Strata",
    "Transaction",
    "setup",
    # Exception hierarchy
    "StrataError",
    "NotFoundError",
    "ValidationError",
    "ConflictError",
    "StateError",
    "ConstraintError",
    "AccessDeniedError",
    "IoError",
]
__version__ = "0.6.0"
