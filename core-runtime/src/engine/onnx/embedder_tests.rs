//! Embedder tests (B-ONNX-1). The fixture-backed tests run against the tiny
//! committed ONNX model (`tests/fixtures/models/onnx/tiny-embedder/`, ~1 KB,
//! generated once by `scripts/gen_onnx_fixture.py` — nothing is downloaded at
//! test time). Golden values were computed by the same generator in float32.

use super::*;
use crate::engine::Model;

const DIM: usize = 8;
#[cfg(feature = "onnx")]
const MODEL_ID: &str = "tiny-embedder";

/// float32 reference: gather rows [3, 4] → masked mean → L2 normalize
/// (`scripts/gen_onnx_fixture.py` prints this as the "hello world" golden;
/// digits kept verbatim from that output, hence the precision allow).
#[cfg(feature = "onnx")]
#[allow(clippy::excessive_precision)]
const GOLDEN_HELLO_WORLD: [f32; DIM] = [
    0.31656653,
    -0.28370523,
    -0.46834815,
    0.033140063,
    0.48607799,
    0.22691055,
    -0.36468142,
    -0.42201424,
];

#[cfg(feature = "onnx")]
fn fixture_model_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/models/onnx")
}

#[cfg(feature = "onnx")]
fn load_fixture() -> OnnxEmbedder {
    OnnxEmbedder::load(&fixture_model_dir(), MODEL_ID, DIM).expect("committed fixture must load")
}

#[cfg(feature = "onnx")]
async fn embed_one(embedder: &OnnxEmbedder, text: &str) -> Vec<f32> {
    let input = InferenceInput::Text(text.to_string());
    match embedder.infer(&input, &InferenceConfig::default()).await {
        Ok(InferenceOutput::Embedding(r)) => r.vector,
        other => panic!("expected Embedding output, got {other:?}"),
    }
}

#[cfg(feature = "onnx")]
#[tokio::test]
async fn golden_vector_hello_world() {
    let embedder = load_fixture();
    let vector = embed_one(&embedder, "hello world").await;
    assert_eq!(vector.len(), DIM);
    for (i, (got, want)) in vector.iter().zip(GOLDEN_HELLO_WORLD.iter()).enumerate() {
        assert!((got - want).abs() < 1e-5, "dim {i}: got {got}, want {want}");
    }
}

#[cfg(feature = "onnx")]
#[tokio::test]
async fn embedding_is_l2_normalized() {
    let embedder = load_fixture();
    let vector = embed_one(&embedder, "greatest good").await;
    let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-5, "L2 norm = {norm}");
}

#[cfg(feature = "onnx")]
#[tokio::test]
async fn deterministic_across_two_loads() {
    let a = embed_one(&load_fixture(), "hello greatest world").await;
    let b = embed_one(&load_fixture(), "hello greatest world").await;
    assert_eq!(a, b, "two independent loads must embed identically");
}

#[cfg(feature = "onnx")]
#[tokio::test]
async fn batch_embeds_every_item_and_matches_per_item() {
    let embedder = load_fixture();
    let texts = ["hello world", "greatest good", "hello"];

    let batch = InferenceInput::TextBatch(texts.iter().map(|t| t.to_string()).collect());
    let results = match embedder.infer(&batch, &InferenceConfig::default()).await {
        Ok(InferenceOutput::EmbeddingBatch(rs)) => rs,
        other => panic!("expected EmbeddingBatch output, got {other:?}"),
    };
    assert_eq!(results.len(), texts.len(), "one result per batch item");

    for (text, result) in texts.iter().zip(&results) {
        assert_eq!(result.dimensions, DIM);
        let single = embed_one(&embedder, text).await;
        assert_eq!(result.vector, single, "batch item for {text:?} != per-item");
    }
}

#[cfg(feature = "onnx")]
#[tokio::test]
async fn memory_usage_tracks_load_and_unload() {
    let mut embedder = load_fixture();
    // 16x8 f32 table = 512 payload bytes at minimum.
    assert!(embedder.memory_usage() >= 512, "loaded footprint reported");
    embedder.unload().await.expect("unload");
    assert_eq!(embedder.memory_usage(), 0, "unload zeroes the footprint");
    let input = InferenceInput::Text("hello".into());
    assert!(
        embedder
            .infer(&input, &InferenceConfig::default())
            .await
            .is_err(),
        "inference after unload must fail loud"
    );
}

#[cfg(feature = "onnx")]
#[test]
fn load_missing_model_fails_cleanly() {
    let missing = fixture_model_dir().join("does-not-exist");
    let err = OnnxEmbedder::load(&missing, "nope", DIM);
    assert!(matches!(err, Err(InferenceError::ModelError(_))));
}

#[cfg(not(feature = "onnx"))]
#[test]
fn load_without_feature_fails_cleanly() {
    let err = OnnxEmbedder::load(std::path::Path::new("/nonexistent"), "any", DIM);
    assert!(matches!(err, Err(InferenceError::ModelError(_))));
}

#[tokio::test]
async fn not_loaded_embedder_errors_cleanly() {
    // `new()` is the not-loaded state under BOTH feature configurations.
    let embedder = OnnxEmbedder::new("missing".into(), DIM);
    let input = InferenceInput::Text("test".into());
    let result = embedder.infer(&input, &InferenceConfig::default()).await;
    assert!(matches!(result, Err(InferenceError::ModelError(_))));
}

#[test]
fn embedding_dim_is_reported() {
    assert_eq!(OnnxEmbedder::new("m".into(), 384).embedding_dim(), 384);
}
