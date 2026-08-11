# AUDIT REPORT

**Tribunal Date**: 2026-08-11T00:00:00-04:00
**Target**: `docs/plan-onnx-real-inference-2026-08-11.md` (B-ONNX-1 — real ONNX embedding inference)
**Risk Grade**: L2
**Auditor**: The QoreLogic Judge

---

## VERDICT: PASS

---

### Executive Summary

The blueprint replaces the ONNX embedder's degraded inference path (unmasked mean pool, no
normalization, nondeterministic output pick, first-item-only batches, zero memory accounting)
with a correct, deterministic pipeline plus a `load()` entry point, using only dependencies
already declared behind the `onnx` feature. No security surface (`ipc/`, `sandbox/`,
`security/`) is touched; model IO is read-only from a caller-supplied directory; the committed
~1 KB fixture eliminates any test-time download. File-split plan keeps every file under the
250-line Razor ceiling. No violation found in any pass.

### Audit Results

#### Security Pass

**Result**: PASS
No auth logic, no credentials, no bypassed checks. `load()` is fail-closed (missing/invalid
`model.onnx` → `InferenceError::ModelError`; not-loaded embedder still errors cleanly).
Tokenizer resolution reuses the existing B-28 offline `from_file` path — no network surface
exists in the dependency set (`candle-onnx`, `candle-core`, `tokenizers` with
`default-features = false`, `http`/`hf-hub` off). Read boundary respected: reads
`<model_dir>/<model_id>/{model.onnx,tokenizer.json}` only, writes nothing.

#### Ghost UI Pass

**Result**: PASS
No UI. Every proposed API lands on real logic: `load` → `with_model` → eval pipeline;
`TextBatch` → per-item eval → `EmbeddingBatch`. No placeholder returns remain on the
embedding path under the `onnx` feature; non-`onnx` builds fail loud as today.

#### Section 4 Razor Pass

| Check              | Limit | Blueprint Proposes | Status |
| ------------------ | ----- | ------------------ | ------ |
| Max function lines | 40    | ≤40 (pipeline split into helpers: select/pool/normalize/estimate) | OK |
| Max file lines     | 250   | embedder.rs ≤250 after test extraction; tensor_ops.rs ≤250; embedder_tests.rs ≤250 | OK |
| Max nesting depth  | 3     | ≤3 (flat helper chain, `?` propagation) | OK |
| Nested ternaries   | 0     | 0 | OK |

**Result**: PASS — the split is mandatory, not optional: embedder.rs sits at 222 lines today
and cannot absorb the additions in place.

#### Dependency Pass

| Package | Justification | <10 Lines Vanilla? | Verdict |
| ------- | ------------- | ------------------ | ------- |
| (none new) | `candle-core`/`candle-onnx`/`tokenizers` already declared, feature-gated | n/a | PASS |

**Result**: PASS — zero new dependencies; forbidden list (`reqwest`, `hyper`, WebSocket,
traversal crates) untouched.

#### Orphan Pass

| Proposed File | Entry Point Connection | Status |
| ------------- | ---------------------- | ------ |
| `engine/onnx/tensor_ops.rs` | `embedder.rs` + `classifier.rs` → `onnx/mod.rs` → `engine` → `lib.rs` | Connected |
| `engine/onnx/embedder_tests.rs` | `#[cfg(test)] #[path]` include from `embedder.rs` (existing dispatch pattern) | Connected |
| `tests/fixtures/models/onnx/tiny-embedder/*` | Referenced by `embedder_tests.rs` via `CARGO_MANIFEST_DIR` | Connected |
| `scripts/gen_onnx_fixture.py` | Dev-only provenance/regeneration script (precedent: `scripts/download-models.sh`); never on build/test path | Connected (tooling) |

**Result**: PASS

#### Macro-Level Architecture Pass

**Result**: PASS
Change confined to `engine/onnx/`; module boundaries and layering unchanged. Shared tensor
helpers get a single source of truth (`tensor_ops.rs`) instead of classifier reaching into
embedder internals — duplication decreases. `InferenceOutput::EmbeddingBatch` is additive;
the only in-repo `match` on `InferenceOutput` (`engine/inference.rs:206`) carries a wildcard
arm. Deterministic output selection mirrors the classifier's existing `logits` rule —
consistent idiom, no new pattern.

### Violations Found

| ID  | Category | Location | Description |
| --- | -------- | -------- | ----------- |
| —   | none     | —        | —           |

### Required Remediation

None. Gate is OPEN.

### Verdict Hash

SHA256(this_report) = recorded in META_LEDGER Entry #203 (Content Hash)

---

_This verdict is binding. Implementation may proceed without modification._
