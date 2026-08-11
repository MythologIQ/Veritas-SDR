//! Shared tensor helpers for the ONNX backend (feature `onnx` only).
//!
//! Single source of truth for the embedder's and classifier's input assembly
//! plus the embedding post-processing pipeline: deterministic output
//! selection, attention-mask-weighted mean pooling, and L2 normalization.

use std::collections::HashMap;

use candle_core::{DType, Device, Tensor};

use crate::engine::InferenceError;

/// Build `input_ids`, `attention_mask`, `token_type_ids` tensors for a single
/// unpadded sequence (`[1, seq_len]`, mask all ones).
pub(super) fn build_transformer_inputs(
    tokens: &[i64],
    device: &Device,
) -> Result<HashMap<String, Tensor>, InferenceError> {
    let ids = Tensor::new(tokens, device)
        .and_then(|t| t.unsqueeze(0))
        .map_err(|e| InferenceError::ModelError(format!("input: {e}")))?;

    let attn =
        Tensor::ones_like(&ids).map_err(|e| InferenceError::ModelError(format!("attn: {e}")))?;

    let ttype =
        Tensor::zeros_like(&ids).map_err(|e| InferenceError::ModelError(format!("ttype: {e}")))?;

    let mut map = HashMap::new();
    map.insert("input_ids".to_string(), ids);
    map.insert("attention_mask".to_string(), attn);
    map.insert("token_type_ids".to_string(), ttype);
    Ok(map)
}

/// Select the hidden-state output tensor deterministically: prefer the named
/// `last_hidden_state`, else accept a single-output model. Fail loud on
/// ambiguity rather than picking a nondeterministic `HashMap` entry (mirrors
/// the classifier's `logits` rule).
pub(super) fn select_hidden_state(
    outputs: &HashMap<String, Tensor>,
) -> Result<&Tensor, InferenceError> {
    outputs
        .get("last_hidden_state")
        .or_else(|| (outputs.len() == 1).then(|| outputs.values().next().unwrap()))
        .ok_or_else(|| {
            InferenceError::ModelError(
                "ambiguous embedder outputs: expected a `last_hidden_state` output \
                 or exactly one output"
                    .into(),
            )
        })
}

/// Mean-pool hidden states over the attention mask.
///
/// `hidden` is `[1, seq, dim]`, `attention_mask` is `[1, seq]`; masked
/// positions contribute nothing and the sum is divided by the number of
/// attended tokens. Returns the pooled `[dim]` vector.
pub(super) fn masked_mean_pool(
    hidden: &Tensor,
    attention_mask: &Tensor,
) -> candle_core::Result<Tensor> {
    let mask = attention_mask.to_dtype(DType::F32)?.unsqueeze(2)?; // [1, seq, 1]
    let summed = hidden.to_dtype(DType::F32)?.broadcast_mul(&mask)?.sum(1)?; // [1, dim]
    let counts = mask.sum(1)?; // [1, 1]
    summed.broadcast_div(&counts)?.squeeze(0) // [dim]
}

/// L2-normalize a vector to unit length. A zero vector is returned unchanged
/// (there is no meaningful direction to normalize to).
pub(super) fn l2_normalize(vector: Vec<f32>) -> Vec<f32> {
    let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 {
        vector.into_iter().map(|v| v / norm).collect()
    } else {
        vector
    }
}

/// Approximate in-memory footprint of a loaded `ModelProto`: the sum of all
/// graph initializer payloads (weights dominate; graph structure is noise).
pub(super) fn model_memory_estimate(model: &candle_onnx::onnx::ModelProto) -> usize {
    model
        .graph
        .as_ref()
        .map(|g| g.initializer.iter().map(tensor_payload_bytes).sum())
        .unwrap_or(0)
}

/// Payload size of one initializer tensor across the ONNX storage variants.
fn tensor_payload_bytes(t: &candle_onnx::onnx::TensorProto) -> usize {
    t.raw_data.len()
        + t.float_data.len() * 4
        + t.int32_data.len() * 4
        + t.int64_data.len() * 8
        + t.double_data.len() * 8
        + t.uint64_data.len() * 8
        + t.string_data.iter().map(|s| s.len()).sum::<usize>()
}
