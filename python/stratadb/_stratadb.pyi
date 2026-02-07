"""Type stubs for the native _stratadb extension module (PyO3)."""

from typing import Any, Optional

import numpy as np
import numpy.typing as npt

# =============================================================================
# Exception hierarchy
# =============================================================================

class StrataError(Exception):
    """Base exception for all StrataDB errors."""
    ...

class NotFoundError(StrataError):
    """Entity not found."""
    ...

class ValidationError(StrataError):
    """Invalid input or type mismatch."""
    ...

class ConflictError(StrataError):
    """Version or concurrency conflict."""
    ...

class StateError(StrataError):
    """Invalid state transition."""
    ...

class ConstraintError(StrataError):
    """Constraint or limit violation."""
    ...

class AccessDeniedError(StrataError):
    """Access denied (read-only)."""
    ...

class IoError(StrataError):
    """I/O, serialization, or internal error."""
    ...

# =============================================================================
# Value type alias
#
# StrataDB values map to Python as follows:
#   Null   → None
#   Bool   → bool
#   Int    → int
#   Float  → float
#   String → str
#   Bytes  → bytes
#   Array  → list
#   Object → dict
# =============================================================================

Value = None | bool | int | float | str | bytes | list[Any] | dict[str, Any]

# =============================================================================
# Main database class
# =============================================================================

class Strata:
    """StrataDB database handle.

    All operations go through this class. Construct via the ``open`` or
    ``cache`` static methods.
    """

    # -- Construction ---------------------------------------------------------

    @staticmethod
    def open(
        path: str,
        auto_embed: bool = False,
        read_only: bool = False,
    ) -> "Strata":
        """Open a database at the given path.

        Args:
            path: Directory path for the database.
            auto_embed: Enable automatic text embedding for semantic search.
            read_only: Open in read-only mode.
        """
        ...

    @staticmethod
    def cache() -> "Strata":
        """Create an in-memory database (no persistence)."""
        ...

    @staticmethod
    def setup() -> str:
        """Download model files for auto-embedding.

        Returns:
            The path where model files are stored.

        Raises:
            RuntimeError: If the download fails or the ``embed`` feature is
                not enabled.
        """
        ...

    # -- KV Store -------------------------------------------------------------

    def kv_put(self, key: str, value: Value) -> int:
        """Store a key-value pair. Returns the version number."""
        ...

    def kv_get(self, key: str) -> Value | None:
        """Get a value by key. Returns ``None`` if not found."""
        ...

    def kv_delete(self, key: str) -> bool:
        """Delete a key. Returns ``True`` if the key existed."""
        ...

    def kv_list(self, prefix: Optional[str] = None) -> list[str]:
        """List keys with optional prefix filter."""
        ...

    def kv_history(self, key: str) -> list[dict[str, Any]] | None:
        """Get version history for a key.

        Returns a list of ``{"value", "version", "timestamp"}`` dicts,
        or ``None`` if the key does not exist.
        """
        ...

    def kv_get_versioned(self, key: str) -> dict[str, Any] | None:
        """Get a value with version info.

        Returns ``{"value", "version", "timestamp"}`` or ``None``.
        """
        ...

    def kv_list_paginated(
        self,
        prefix: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> dict[str, Any]:
        """List keys with pagination support.

        Returns ``{"keys": [...]}``.

        Note:
            Cursor-based pagination is not yet supported for KV lists.
        """
        ...

    # -- State Cell -----------------------------------------------------------

    def state_set(self, cell: str, value: Value) -> int:
        """Set a state cell value. Returns the version number."""
        ...

    def state_get(self, cell: str) -> Value | None:
        """Get a state cell value. Returns ``None`` if not found."""
        ...

    def state_init(self, cell: str, value: Value) -> int:
        """Initialize a state cell if it doesn't already exist."""
        ...

    def state_cas(
        self,
        cell: str,
        new_value: Value,
        expected_version: Optional[int] = None,
    ) -> int | None:
        """Compare-and-swap update based on version.

        Returns the new version on success, ``None`` on CAS failure.
        """
        ...

    def state_history(self, cell: str) -> list[dict[str, Any]] | None:
        """Get version history for a state cell."""
        ...

    def state_delete(self, cell: str) -> bool:
        """Delete a state cell. Returns ``True`` if it existed."""
        ...

    def state_list(self, prefix: Optional[str] = None) -> list[str]:
        """List state cell names with optional prefix filter."""
        ...

    def state_get_versioned(self, cell: str) -> dict[str, Any] | None:
        """Get a state cell value with version info."""
        ...

    # -- Event Log ------------------------------------------------------------

    def event_append(self, event_type: str, payload: Value) -> int:
        """Append an event to the log. Returns the sequence number."""
        ...

    def event_get(self, sequence: int) -> dict[str, Any] | None:
        """Get an event by sequence number."""
        ...

    def event_list(self, event_type: str) -> list[dict[str, Any]]:
        """List events by type."""
        ...

    def event_len(self) -> int:
        """Get total event count."""
        ...

    def event_list_paginated(
        self,
        event_type: str,
        limit: Optional[int] = None,
        after: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """List events by type with pagination support."""
        ...

    # -- JSON Store -----------------------------------------------------------

    def json_set(self, key: str, path: str, value: Value) -> int:
        """Set a value at a JSONPath. Returns the version number."""
        ...

    def json_get(self, key: str, path: str) -> Value | None:
        """Get a value at a JSONPath."""
        ...

    def json_delete(self, key: str, path: str) -> int:
        """Delete a JSON document. Returns the version number."""
        ...

    def json_history(self, key: str) -> list[dict[str, Any]] | None:
        """Get version history for a JSON document."""
        ...

    def json_list(
        self,
        limit: int,
        prefix: Optional[str] = None,
        cursor: Optional[str] = None,
    ) -> dict[str, Any]:
        """List JSON document keys.

        Returns ``{"keys": [...], "cursor": ...}`` where cursor is present
        only when more pages are available.
        """
        ...

    def json_get_versioned(self, key: str) -> dict[str, Any] | None:
        """Get a JSON document value with version info."""
        ...

    # -- Vector Store ---------------------------------------------------------

    def vector_create_collection(
        self,
        collection: str,
        dimension: int,
        metric: Optional[str] = None,
    ) -> int:
        """Create a vector collection.

        Args:
            collection: Collection name.
            dimension: Vector dimensionality.
            metric: Distance metric (``"cosine"``, ``"euclidean"``, or
                ``"dot_product"``). Defaults to ``"cosine"``.
        """
        ...

    def vector_delete_collection(self, collection: str) -> bool:
        """Delete a vector collection."""
        ...

    def vector_list_collections(self) -> list[dict[str, Any]]:
        """List vector collections.

        Each dict contains ``name``, ``dimension``, ``metric``, ``count``,
        ``index_type``, ``memory_bytes``.
        """
        ...

    def vector_upsert(
        self,
        collection: str,
        key: str,
        vector: npt.NDArray[np.float32] | list[float],
        metadata: Value | None = None,
    ) -> int:
        """Insert or update a vector. Returns the version number."""
        ...

    def vector_get(self, collection: str, key: str) -> dict[str, Any] | None:
        """Get a vector by key.

        Returns ``{"key", "embedding", "metadata", "version", "timestamp"}``
        or ``None``.
        """
        ...

    def vector_delete(self, collection: str, key: str) -> bool:
        """Delete a vector."""
        ...

    def vector_search(
        self,
        collection: str,
        query: npt.NDArray[np.float32] | list[float],
        k: int,
    ) -> list[dict[str, Any]]:
        """Search for similar vectors.

        Returns a list of ``{"key", "score", "metadata"}`` dicts.
        """
        ...

    def vector_collection_stats(self, collection: str) -> dict[str, Any]:
        """Get statistics for a single collection."""
        ...

    def vector_batch_upsert(
        self,
        collection: str,
        vectors: list[dict[str, Any]],
    ) -> list[int]:
        """Batch insert/update multiple vectors.

        Each vector dict must have ``"key"`` and ``"vector"`` keys, with
        optional ``"metadata"``.
        """
        ...

    def vector_search_filtered(
        self,
        collection: str,
        query: npt.NDArray[np.float32] | list[float],
        k: int,
        metric: Optional[str] = None,
        filter: Optional[list[dict[str, Any]]] = None,
    ) -> list[dict[str, Any]]:
        """Search for similar vectors with optional filter and metric override."""
        ...

    # -- Branch Management ----------------------------------------------------

    def current_branch(self) -> str:
        """Get the current branch name."""
        ...

    def set_branch(self, branch: str) -> None:
        """Switch to a different branch."""
        ...

    def create_branch(self, branch: str) -> None:
        """Create a new empty branch."""
        ...

    def fork_branch(self, destination: str) -> dict[str, Any]:
        """Fork the current branch, copying all data.

        Returns ``{"source", "destination", "keys_copied"}``.
        """
        ...

    def list_branches(self) -> list[str]:
        """List all branches."""
        ...

    def delete_branch(self, branch: str) -> None:
        """Delete a branch."""
        ...

    def branch_exists(self, name: str) -> bool:
        """Check if a branch exists."""
        ...

    def branch_get(self, name: str) -> dict[str, Any] | None:
        """Get branch metadata with version info.

        Returns ``{"id", "status", "created_at", "updated_at", "version",
        "timestamp"}`` or ``None``.
        """
        ...

    def diff_branches(
        self, branch_a: str, branch_b: str
    ) -> dict[str, Any]:
        """Compare two branches.

        Returns ``{"branch_a", "branch_b", "summary": {"total_added",
        "total_removed", "total_modified"}}``.
        """
        ...

    def merge_branches(
        self,
        source: str,
        strategy: Optional[str] = None,
    ) -> dict[str, Any]:
        """Merge a branch into the current branch.

        Args:
            source: Branch to merge from.
            strategy: ``"last_writer_wins"`` (default) or ``"strict"``.

        Returns ``{"keys_applied", "spaces_merged", "conflicts"}``.
        """
        ...

    # -- Space Management -----------------------------------------------------

    def current_space(self) -> str:
        """Get the current space name."""
        ...

    def set_space(self, space: str) -> None:
        """Switch to a different space."""
        ...

    def list_spaces(self) -> list[str]:
        """List all spaces in the current branch."""
        ...

    def delete_space(self, space: str) -> None:
        """Delete a space and all its data."""
        ...

    def delete_space_force(self, space: str) -> None:
        """Force delete a space even if non-empty."""
        ...

    def space_create(self, space: str) -> None:
        """Create a new space explicitly."""
        ...

    def space_exists(self, space: str) -> bool:
        """Check if a space exists in the current branch."""
        ...

    # -- Database Operations --------------------------------------------------

    def ping(self) -> str:
        """Check database connectivity."""
        ...

    def info(self) -> dict[str, Any]:
        """Get database info.

        Returns ``{"version", "uptime_secs", "branch_count", "total_keys"}``.
        """
        ...

    def flush(self) -> None:
        """Flush writes to disk."""
        ...

    def compact(self) -> None:
        """Trigger compaction."""
        ...

    # -- Bundle Operations ----------------------------------------------------

    def branch_export(self, branch: str, path: str) -> dict[str, Any]:
        """Export a branch to a bundle file.

        Returns ``{"branch_id", "path", "entry_count", "bundle_size"}``.
        """
        ...

    def branch_import(self, path: str) -> dict[str, Any]:
        """Import a branch from a bundle file.

        Returns ``{"branch_id", "transactions_applied", "keys_written"}``.
        """
        ...

    def branch_validate_bundle(self, path: str) -> dict[str, Any]:
        """Validate a bundle file without importing.

        Returns ``{"branch_id", "format_version", "entry_count",
        "checksums_valid"}``.
        """
        ...

    # -- Transaction Operations -----------------------------------------------

    def begin(self, read_only: Optional[bool] = None) -> None:
        """Begin a new transaction.

        Raises:
            StateError: If a transaction is already active.
        """
        ...

    def commit(self) -> int:
        """Commit the current transaction. Returns the commit version."""
        ...

    def rollback(self) -> None:
        """Rollback the current transaction."""
        ...

    def txn_info(self) -> dict[str, Any] | None:
        """Get current transaction info.

        Returns ``{"id", "status", "started_at"}`` or ``None``.
        """
        ...

    def txn_is_active(self) -> bool:
        """Check if a transaction is currently active."""
        ...

    # -- Search ---------------------------------------------------------------

    def search(
        self,
        query: str,
        k: Optional[int] = None,
        primitives: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """Search across multiple primitives for matching content.

        Returns a list of ``{"entity", "primitive", "score", "rank",
        "snippet"}`` dicts.
        """
        ...

    # -- Retention ------------------------------------------------------------

    def retention_apply(self) -> None:
        """Apply retention policy (trigger garbage collection)."""
        ...
