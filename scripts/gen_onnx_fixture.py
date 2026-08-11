#!/usr/bin/env python3
"""Generate the tiny ONNX embedder test fixture (B-ONNX-1).

Writes core-runtime/tests/fixtures/models/onnx/tiny-embedder/{model.onnx,
tokenizer.json}: a ~1 KB Gather-over-constant-table graph shaped like a
transformer encoder (inputs input_ids/attention_mask/token_type_ids [batch,
seq] i64, output last_hidden_state [batch, seq, 8] f32) plus a 7-token
WordPiece tokenizer. Deterministic by construction — rerunning reproduces
byte-identical semantics (table values are a fixed sin() formula).

Dev-only provenance/regeneration tool. It is NEVER run at build or test
time; the generated files are committed. Requires: pip install onnx numpy.
Also prints the golden vector asserted by embedder_tests.rs.
"""

import json
import math
import pathlib

import numpy as np
import onnx
from onnx import TensorProto, helper

VOCAB, DIM = 16, 8
OUT_DIR = (
    pathlib.Path(__file__).resolve().parent.parent
    / "core-runtime/tests/fixtures/models/onnx/tiny-embedder"
)


def build_table() -> np.ndarray:
    """Fixed, non-degenerate embedding table: table[i][j] = sin(0.7*i + 1.3*j)."""
    return np.array(
        [[math.sin(0.7 * i + 1.3 * j) for j in range(DIM)] for i in range(VOCAB)],
        dtype=np.float32,
    )


def build_model(table: np.ndarray) -> onnx.ModelProto:
    init = helper.make_tensor(
        "embeddings", TensorProto.FLOAT, [VOCAB, DIM], table.flatten().tolist()
    )
    inputs = [
        helper.make_tensor_value_info(name, TensorProto.INT64, ["batch", "seq"])
        for name in ("input_ids", "attention_mask", "token_type_ids")
    ]
    output = helper.make_tensor_value_info(
        "last_hidden_state", TensorProto.FLOAT, ["batch", "seq", DIM]
    )
    gather = helper.make_node(
        "Gather", ["embeddings", "input_ids"], ["last_hidden_state"], axis=0
    )
    graph = helper.make_graph([gather], "tiny_embedder", inputs, [output], [init])
    model = helper.make_model(
        graph,
        producer_name="gg-core-fixture-gen",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def tokenizer_json() -> dict:
    vocab = {"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "hello": 3, "world": 4,
             "greatest": 5, "good": 6}
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": None,
        "decoder": None,
        "model": {
            "type": "WordPiece",
            "unk_token": "[UNK]",
            "continuing_subword_prefix": "##",
            "max_input_chars_per_word": 100,
            "vocab": vocab,
        },
    }


def golden(table: np.ndarray, ids: list[int]) -> np.ndarray:
    """Reference pipeline in float32: gather -> masked mean -> L2 normalize."""
    hidden = table[ids]  # mask is all ones for an unpadded single sequence
    pooled = hidden.sum(axis=0, dtype=np.float32) / np.float32(len(ids))
    norm = np.float32(math.sqrt(float((pooled * pooled).sum(dtype=np.float32))))
    return pooled / norm


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table = build_table()
    onnx.save(build_model(table), OUT_DIR / "model.onnx")
    (OUT_DIR / "tokenizer.json").write_text(
        json.dumps(tokenizer_json(), indent=2) + "\n"
    )
    print(f"wrote {OUT_DIR}/model.onnx + tokenizer.json")
    # "hello world" -> WordPiece ids [3, 4] (no post-processor => no CLS/SEP)
    vec = golden(table, [3, 4])
    print("golden 'hello world':", ", ".join(f"{v:.8}" for v in vec))
    print("l2 norm:", float(np.sqrt((vec * vec).sum())))


if __name__ == "__main__":
    main()
