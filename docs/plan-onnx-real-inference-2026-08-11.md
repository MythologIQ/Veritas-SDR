# Plan: Real ONNX Embedding Inference (B-ONNX-1)

**Date**: 2026-08-11
**Risk Grade**: L2 (engine logic change; no auth/security surface touched)
**Branch**: `feature/onnx-real-inference`
**Consumer**: EvolveAI plan-v6.2 Phase 1 (`ggcore` cargo feature, `vendor/GG-CORE` submodule) —
its adapter calls `embedder.infer(&InferenceInput::Text(...), &config)` and expects
`InferenceOutput::Embedding`.

## Problem

`OnnxEmbedder` cannot be constructed from model files by an external caller: there is no
`load()` API. Pooling is an unmasked mean with no L2 normalization, output-tensor selection
is nondeterministic (`HashMap::values().next()`), `TextBatch` embeds only the first item,
and `memory_usage()` always reports 0.

## Blueprint

### B1. `OnnxEmbedder::load(model_dir: &Path, model_id: &str, embedding_dim: usize)`

- Reads `<model_dir>/<model_id>/model.onnx` via `candle_onnx::read_file` (local disk only,
  read-only — respects the `models/` read boundary; no network path exists in the deps).
- Tokenizer resolved from sibling `tokenizer.json` via existing `OnnxTokenizer::for_model`
  (offline `from_file`; degrades loud to hash fallback per existing B-28 convention).
- Fail-closed: missing/invalid `model.onnx` → `InferenceError::ModelError`.
- Compiled under both feature configurations: without `onnx` it returns `ModelError`
  ("onnx feature not enabled") so downstream code compiles either way.
- Existing `new()` stays as the not-loaded state; inference on it still errors cleanly.

### B2. Correct pooling pipeline (feature `onnx`)

tokenize → `simple_eval` → select hidden-state output **deterministically**
(prefer `last_hidden_state`, else the single output, else fail loud — mirrors the
classifier's `logits` rule) → **masked mean-pool** over the attention mask →
**L2-normalize** → `EmbeddingResult`. Target model is all-MiniLM-L6-v2 (384 dims) but
nothing model-specific is hardcoded beyond the existing `DEFAULT_EMBEDDING_DIM` default.

### B3. Batch semantics

`InferenceInput::TextBatch(batch)` embeds **every** item (sequential per-item eval, batch
size already capped at 32 by input validation) and returns a new **additive** variant
`InferenceOutput::EmbeddingBatch(Vec<EmbeddingResult>)`. `InferenceInput::Text` still
returns `InferenceOutput::Embedding` (EvolveAI source compatibility). The only in-repo
`match` on `InferenceOutput` (`engine/inference.rs`) has a wildcard arm — no breakage.

### B4. Memory accounting

`with_model` estimates the session footprint by summing initializer payloads in the
`ModelProto` graph (raw + typed tensor data); `memory_usage()` reports it; `unload()`
drops the model and zeroes the count (existing behavior, now meaningful).

### B5. File layout (Section 4 Razor)

`embedder.rs` is at 222 lines and cannot absorb this. Split:

| File | Content | Budget |
|---|---|---|
| `engine/onnx/embedder.rs` | struct, `new`/`load`/`with_model`, infer/Model impl | ≤250 |
| `engine/onnx/tensor_ops.rs` (new) | `build_transformer_inputs` (moved), output selection, masked mean-pool, L2-normalize, initializer-size estimate | ≤250 |
| `engine/onnx/embedder_tests.rs` (new) | unit tests via `#[path]` (existing dispatch/classifier pattern) | ≤250 |

`classifier.rs` import of `build_transformer_inputs` updates to `tensor_ops`.
All functions ≤40 lines, nesting ≤3.

### B6. Test fixture (committed, zero downloads)

A tiny handcrafted ONNX graph committed at
`core-runtime/tests/fixtures/models/onnx/tiny-embedder/{model.onnx,tokenizer.json}`
(~1 KB): inputs `input_ids`/`attention_mask`/`token_type_ids` `[batch, seq]` i64, one
`Gather` over a constant 16×8 embedding table → `last_hidden_state` `[batch, seq, 8]`.
The WordPiece `tokenizer.json` carries a 7-token vocab. The generator script is committed
at `scripts/gen_onnx_fixture.py` for provenance and regeneration; it is **never** run at
build or test time. Tests:

1. Golden vector: embed known text, compare to hardcoded expected floats (1e-5).
2. L2 norm == 1.0 within 1e-5.
3. Determinism: two independent `load()`s produce identical vectors.
4. `TextBatch` == per-item `Text` results, one result per item.
5. Not-loaded embedder (`new()`) errors cleanly (both feature configs).
6. `load()` on a missing directory errors cleanly.
7. `memory_usage()` > 0 after load, == 0 after `unload()`.

## Constraint compliance

- No new dependencies. `candle-onnx`/`candle-core`/`tokenizers` already declared behind
  the `onnx` feature; no `reqwest`/`hyper`/WebSocket/traversal crates.
- Read-only model IO under a caller-supplied directory; write nothing.
- No IPC/network surface change; no forbidden modules.

## Gates

`cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`, `cargo test` — each with
and without `--features onnx`.
