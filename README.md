# strata-python

Python SDK for [StrataDB](https://github.com/strata-systems/strata-core) - an embedded database for AI agents.

PyO3-based bindings embedding the Rust library directly in a Python native extension. No network hop, no serialization overhead beyond the Python/Rust boundary.

## Installation

### From PyPI (coming soon)

```bash
pip install stratadb
```

### From Source

Requires Rust toolchain and maturin:

```bash
git clone https://github.com/strata-systems/strata-python.git
cd strata-python
pip install maturin
maturin develop
```

## Quick Start

```python
from stratadb import Strata

# Open a database (or use Strata.cache() for in-memory)
db = Strata.open("/path/to/data")

# Key-value storage
db.kv_put("user:123", "Alice")
print(db.kv_get("user:123"))  # "Alice"

# Branch isolation (like git branches)
db.create_branch("experiment")
db.set_branch("experiment")
print(db.kv_get("user:123"))  # None - isolated

# Space organization within branches
db.set_space("conversations")
db.kv_put("msg_001", "hello")
```

## Features

### Six Data Primitives

| Primitive | Purpose | Key Methods |
|-----------|---------|-------------|
| **KV Store** | Working memory, config | `kv_put`, `kv_get`, `kv_delete`, `kv_list` |
| **Event Log** | Immutable audit trail | `event_append`, `event_get`, `event_list` |
| **State Cell** | CAS-based coordination | `state_set`, `state_get`, `state_cas` |
| **JSON Store** | Structured documents | `json_set`, `json_get`, `json_delete` |
| **Vector Store** | Embeddings, similarity search | `vector_upsert`, `vector_search` |
| **Branch** | Data isolation | `create_branch`, `set_branch`, `fork_branch` |

### Transactions

```python
# Context manager (recommended) — auto-commits on success, auto-rollbacks on exception
with db.transaction():
    db.kv_put("key1", "value1")
    db.kv_put("key2", "value2")

# Read-only transaction
with db.transaction(read_only=True):
    value = db.kv_get("key1")

# Manual control
db.begin()
db.kv_put("a", 1)
db.commit()  # or db.rollback()
```

### Search

```python
# Hybrid search across all primitives (KV, JSON, Event, State, Branch, Vector)
results = db.search("weather forecast", k=10)
for hit in results:
    print(f"[{hit['primitive']}] {hit['entity']} (score: {hit['score']:.3f})")

# Filter to specific primitives
results = db.search("config", primitives=["kv", "json"])

# Vector search with metadata filters
results = db.vector_search_filtered(
    "embeddings", query_vector, k=10,
    metric="cosine",
    filter=[{"field": "category", "op": "eq", "value": "science"}],
)
```

### NumPy Integration

Vector operations accept and return NumPy arrays:

```python
import numpy as np

# Create a collection
db.vector_create_collection("embeddings", 384, metric="cosine")

# Upsert with NumPy array
embedding = np.random.rand(384).astype(np.float32)
db.vector_upsert("embeddings", "doc-1", embedding, metadata={"title": "Hello"})

# Search returns matches with scores
results = db.vector_search("embeddings", embedding, k=10)
for match in results:
    print(f"{match['key']}: {match['score']}")
```

### Branch Operations

```python
# Fork current branch (copies all data)
db.fork_branch("experiment")

# Compare branches
diff = db.diff_branches("default", "experiment")
print(f"Added: {diff['summary']['total_added']}")

# Merge branches
result = db.merge_branches("experiment", strategy="last_writer_wins")
print(f"Keys applied: {result['keys_applied']}")
```

### Event Log

```python
# Append events
db.event_append("tool_call", {"tool": "search", "query": "weather"})
db.event_append("tool_call", {"tool": "calculator", "expr": "2+2"})

# Get by sequence number
event = db.event_get(0)

# List by type
tool_calls = db.event_list("tool_call")

# Paginated listing
page = db.event_list_paginated("tool_call", limit=10, after=5)
```

### Compare-and-Swap (Version-based)

```python
# Initialize if not exists
db.state_init("counter", 0)

# CAS is version-based - pass expected version, not expected value
# Get current value and its version via state_set (returns version)
version = db.state_set("counter", 1)

# Update only if version matches
new_version = db.state_cas("counter", 2, version)  # (cell, new_value, expected_version)
if new_version is None:
    print("CAS failed - version mismatch")
```

### Error Handling

All exceptions inherit from `StrataError`:

```python
from stratadb import StrataError, NotFoundError, ValidationError

try:
    db.kv_get("missing_key")
except NotFoundError:
    print("Key not found")
except StrataError:
    print("Some other database error")
```

| Exception | When raised |
|-----------|-------------|
| `StrataError` | Base class for all errors |
| `NotFoundError` | Entity not found (key, branch, collection, etc.) |
| `ValidationError` | Invalid input or type mismatch |
| `ConflictError` | Version or concurrency conflict |
| `StateError` | Invalid state transition (e.g., duplicate branch) |
| `ConstraintError` | Constraint violation (e.g., dimension mismatch) |
| `AccessDeniedError` | Access denied (read-only mode) |
| `IoError` | I/O, serialization, or internal error |

### Auto-Embedding

Enable automatic text embedding for semantic search:

```python
# Downloads MiniLM-L6-v2 model (~80MB) on first use
db = Strata.open("/path/to/data", auto_embed=True)

# Or pre-download the model
import stratadb
stratadb.setup()
```

## API Reference

### Strata

| Method | Description |
|--------|-------------|
| `Strata.open(path, auto_embed=False, read_only=False)` | Open database at path |
| `Strata.cache()` | Create in-memory database |
| `Strata.setup()` | Download embedding model files |

### KV Store

| Method | Description |
|--------|-------------|
| `kv_put(key, value)` | Store a value (returns version) |
| `kv_get(key)` | Get a value (returns `None` if missing) |
| `kv_delete(key)` | Delete a key |
| `kv_list(prefix=None)` | List keys |
| `kv_history(key)` | Get version history |
| `kv_get_versioned(key)` | Get value with version info |
| `kv_list_paginated(prefix=None, limit=None)` | List keys with limit |

### State Cell

| Method | Description |
|--------|-------------|
| `state_set(cell, value)` | Set value (returns version) |
| `state_get(cell)` | Get value |
| `state_init(cell, value)` | Initialize if not exists |
| `state_cas(cell, new_value, expected_version)` | Compare-and-swap (version-based) |
| `state_history(cell)` | Get version history |
| `state_delete(cell)` | Delete a state cell |
| `state_list(prefix=None)` | List state cell names |
| `state_get_versioned(cell)` | Get value with version info |

### Event Log

| Method | Description |
|--------|-------------|
| `event_append(type, payload)` | Append event (payload must be a dict) |
| `event_get(sequence)` | Get by sequence |
| `event_list(type)` | List by type |
| `event_len()` | Get count |
| `event_list_paginated(type, limit=None, after=None)` | Paginated listing |

### JSON Store

| Method | Description |
|--------|-------------|
| `json_set(key, path, value)` | Set at JSONPath |
| `json_get(key, path)` | Get at JSONPath |
| `json_delete(key, path)` | Delete |
| `json_history(key)` | Get version history |
| `json_list(limit, prefix=None, cursor=None)` | List keys (paginated) |
| `json_get_versioned(key)` | Get value with version info |

### Vector Store

| Method | Description |
|--------|-------------|
| `vector_create_collection(name, dim, metric=None)` | Create collection (`"cosine"`, `"euclidean"`, `"dot_product"`) |
| `vector_delete_collection(name)` | Delete collection |
| `vector_list_collections()` | List collections |
| `vector_upsert(collection, key, vector, metadata=None)` | Insert/update |
| `vector_get(collection, key)` | Get vector |
| `vector_delete(collection, key)` | Delete vector |
| `vector_search(collection, query, k)` | Similarity search |
| `vector_search_filtered(collection, query, k, metric=None, filter=None)` | Search with filters |
| `vector_collection_stats(collection)` | Get collection statistics |
| `vector_batch_upsert(collection, vectors)` | Batch insert/update |

### Search

| Method | Description |
|--------|-------------|
| `search(query, k=None, primitives=None)` | Hybrid search across primitives |

### Branches

| Method | Description |
|--------|-------------|
| `current_branch()` | Get current branch |
| `set_branch(name)` | Switch branch |
| `create_branch(name)` | Create empty branch |
| `fork_branch(dest)` | Fork with data copy |
| `list_branches()` | List all branches |
| `delete_branch(name)` | Delete branch |
| `branch_exists(name)` | Check if branch exists |
| `branch_get(name)` | Get branch metadata |
| `diff_branches(a, b)` | Compare branches |
| `merge_branches(source, strategy=None)` | Merge into current |

### Spaces

| Method | Description |
|--------|-------------|
| `current_space()` | Get current space |
| `set_space(name)` | Switch space |
| `list_spaces()` | List spaces |
| `space_create(name)` | Create a space explicitly |
| `space_exists(name)` | Check if space exists |
| `delete_space(name)` | Delete space |
| `delete_space_force(name)` | Force delete space |

### Transactions

| Method | Description |
|--------|-------------|
| `transaction(read_only=False)` | Context manager (auto-commit/rollback) |
| `begin(read_only=None)` | Begin transaction manually |
| `commit()` | Commit (returns version) |
| `rollback()` | Rollback |
| `txn_info()` | Get transaction info |
| `txn_is_active()` | Check if transaction is active |

### Bundle Operations

| Method | Description |
|--------|-------------|
| `branch_export(branch, path)` | Export branch to bundle file |
| `branch_import(path)` | Import branch from bundle file |
| `branch_validate_bundle(path)` | Validate bundle file |

### Database

| Method | Description |
|--------|-------------|
| `ping()` | Health check |
| `info()` | Get database info |
| `flush()` | Flush to disk |
| `compact()` | Trigger compaction |
| `retention_apply()` | Apply retention policy (garbage collection) |

## Type Stubs

The package includes a `py.typed` marker and `_stratadb.pyi` stub file for full IDE autocompletion and mypy support.

## Limitations

- **Bytes roundtrip via bundles:** `bytes` values survive normal get/put operations, but a bundle export/import cycle serializes through JSON. Byte values are base64-encoded on export and decoded as `str` on import.
- **KV pagination:** `kv_list_paginated` does not support cursor-based pagination. Use `prefix` and `limit` to narrow results.

## Development

```bash
# Install dev dependencies
pip install maturin pytest numpy

# Build and install in development mode
maturin develop

# Run tests
pytest tests/
```

## License

MIT
