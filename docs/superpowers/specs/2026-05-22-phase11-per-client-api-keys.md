# Phase 11: Per-Client API Keys

**Date:** 2026-05-22

---

## Goal

Replace the single global API key on `/analyze` with per-client API keys. Each client gets a unique key generated on creation. Management routes (`/clients`, `/jobs`) keep the global admin key unchanged.

---

## Architecture

Three changes work together:

1. **`ClientConfig.api_key`** — a new `Optional[str]` field stored in `clients.json` as plain text. The file is already the trust boundary; hashing adds complexity with no meaningful gain.

2. **Key lifecycle** — `POST /clients` generates a 64-char hex key via `secrets.token_hex(32)` and returns it once in the creation response. `POST /clients/{client_id}/rotate-key` (protected by the global admin key) generates a new key, stores it immediately invalidating the old one, and returns the new key. Keys are never returned again after creation/rotation.

3. **New auth dependency `make_verify_client_api_key(registry)`** — reads `X-Client-ID` and `X-API-Key` request headers, looks up the client in the registry, raises `401` if the client doesn't exist or the key doesn't match. `/analyze` switches from the global `verify_api_key` to this dependency. All `/clients` and `/jobs` routes keep the existing global `verify_api_key`.

---

## Request Flow

```
POST /analyze
  X-Client-ID: acme
  X-API-Key: a1b2c3d4e5f6...  (64-char hex)

→ verify_client_api_key:
    client = registry.get("acme")
    if client is None → 401 "Client not found"
    if X-API-Key != client.api_key → 401 "Invalid or missing API key"
    → continue to handler
```

```
POST /clients/{client_id}/rotate-key
  X-API-Key: <global admin key>

→ verify_api_key (global)
→ generate new key via secrets.token_hex(32)
→ registry.upsert(client with new api_key)
→ return {"api_key": "<new key>"}
```

---

## Component Designs

### 1. `src/core/models.py`

Add to `ClientConfig`:

```python
api_key: Optional[str] = None
```

---

### 2. `src/core/auth.py`

Add alongside existing `make_verify_api_key`:

```python
def make_verify_client_api_key(registry):
    def verify_client_api_key(
        x_client_id: str = Header(default=""),
        x_api_key: str = Header(default=""),
    ):
        client = registry.get(x_client_id)
        if client is None or client.api_key != x_api_key:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return verify_client_api_key
```

---

### 3. `main.py` changes

**Key generation on client creation** (`POST /clients`):

```python
import secrets

@app.post("/clients", status_code=201, dependencies=[Depends(verify_api_key)])
def create_client(config: ClientConfig):
    api_key = secrets.token_hex(32)
    config = config.model_copy(update={"api_key": api_key})
    registry.upsert(config)
    return {"client_id": config.client_id, "api_key": api_key}
```

**Rotation endpoint** (add after existing `/clients` routes):

```python
@app.post("/clients/{client_id}/rotate-key", dependencies=[Depends(verify_api_key)])
def rotate_client_key(client_id: str):
    config = registry.get(client_id)
    if config is None:
        raise HTTPException(status_code=404, detail="Client not found")
    new_key = secrets.token_hex(32)
    registry.upsert(config.model_copy(update={"api_key": new_key}))
    return {"api_key": new_key}
```

**Wire new dependency to `/analyze`**:

```python
verify_client_api_key = make_verify_client_api_key(registry)

@app.post("/analyze", dependencies=[Depends(verify_client_api_key)])
```

---

## File Structure

**Modified files:**
- `src/core/models.py` — add `api_key: Optional[str] = None` to `ClientConfig`
- `src/core/auth.py` — add `make_verify_client_api_key(registry)` factory
- `main.py` — key generation in `POST /clients`, rotation endpoint, `/analyze` dependency swap

**New test files:**
- `tests/core/test_auth.py` — unit tests for `make_verify_client_api_key`: missing client, wrong key, correct key
- `tests/api/test_clients_key.py` — HTTP tests: key present in creation response, rotation returns new key, `/analyze` rejects wrong key, `/analyze` accepts correct key

---

## Testing Strategy

**`tests/core/test_auth.py`** (unit tests, no HTTP):
- `test_verify_client_api_key_missing_client` — unknown client_id → 401
- `test_verify_client_api_key_wrong_key` — known client, wrong key → 401
- `test_verify_client_api_key_correct_key` — known client, correct key → passes (no exception)

**`tests/api/test_clients_key.py`** (HTTP via `TestClient`):
- `test_create_client_returns_api_key` — `POST /clients` response contains `api_key` field, length == 64
- `test_rotate_key_returns_new_key` — `POST /clients/{id}/rotate-key` returns new 64-char key, different from original
- `test_analyze_rejects_wrong_key` — `/analyze` with wrong `X-API-Key` → 401
- `test_analyze_accepts_correct_key` — `/analyze` with correct `X-Client-ID` + `X-API-Key` → 200

---

## Self-Review

**Placeholder scan:** No TBDs. All code blocks complete.

**Internal consistency:**
- `secrets.token_hex(32)` produces a 64-char hex string. Tests assert `len == 64`. Consistent.
- `make_verify_client_api_key` compares `client.api_key != x_api_key` — both are strings, direct equality. Consistent with how `make_verify_api_key` works.
- `registry.upsert()` replaces the full config — `api_key` survives subsequent upserts as long as the field is present. Consistent with existing upsert behavior.
- Rotation uses `model_copy(update={...})` — same pattern used elsewhere in the codebase.

**Scope check:** Single focused phase. No overlap with Phase 10 (observability) or Phase 9 (connectors).

**Ambiguity check:**
- Clients created before Phase 11 have `api_key: None`. `/analyze` for those clients will always 401 until a key is assigned via rotate-key. This is intentional — old clients are gated until explicitly provisioned.
- The `POST /clients` response currently returns nothing (status 201, no body). This phase changes it to return `{"client_id": ..., "api_key": ...}`. Existing callers ignoring the body are unaffected.
