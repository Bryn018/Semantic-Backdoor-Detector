#!/usr/bin/env python3
"""
infer.py — Semantic CPG Backdoor Detector CLI.

Analyzes a Python file using:
    lightweight CPG -> CodeBERT (768-dim) -> SemanticBackdoorGNN -> GNNExplainer

Exit codes:
    0 — BENIGN
    1 — MALICIOUS
    2 — Error
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import torch
from torch_geometric.data import Data
from torch_geometric.explain import GNNExplainer, ExplainerConfig, ModelConfig

from gnn_model_v2 import SemanticBackdoorGNN
from embedder import embed_graph_nodes, get_embedding_dim, extract_node_text
from lightweight_cpg import generate_cpg, LABEL_UNKNOWN

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_PATH: Path = Path(__file__).parent / "model.pth"
VOCAB_PATH: Path = Path(__file__).parent / "vocab.json"
THRESHOLD: float = 0.5

ACTIONABLE_TYPES = {
    "CALL",
    "IDENTIFIER",
    "LITERAL",
    "METHOD_PARAMETER_IN",
    "CONTROL_STRUCTURE",
}

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger("infer")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_graph(py_file: Path) -> dict:
    cpg = generate_cpg(py_file.read_text(encoding="utf-8"))
    if not cpg.get("nodes"):
        raise RuntimeError("CPG generation produced no nodes")
    return cpg


def _build_data(graph: dict, embed_dim: int) -> Data:
    index_to_code: dict[int, str] = {}
    index_to_label: dict[int, str] = {}
    id_map: dict[int, int] = {}

    for idx, node in enumerate(graph["nodes"]):
        nid = int(node.get("id", idx))
        id_map[nid] = idx
        index_to_code[idx] = extract_node_text(node) or node.get("LABEL", LABEL_UNKNOWN)
        index_to_label[idx] = node.get("label", LABEL_UNKNOWN)

    src_list: list[int] = []
    dst_list: list[int] = []
    for edge in graph["edges"]:
        s = int(edge.get("src", -1))
        d = int(edge.get("dst", -1))
        if s in id_map and d in id_map:
            src_list.append(id_map[s])
            dst_list.append(id_map[d])

    if not src_list:
        src_list = list(range(len(graph["nodes"])))
        dst_list = list(range(len(graph["nodes"])))

    x_np = embed_graph_nodes(graph)
    x = torch.from_numpy(x_np).float()
    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=torch.tensor([0], dtype=torch.long)), index_to_code, index_to_label


def _explain(data: Data, model: SemanticBackdoorGNN, index_to_code: dict, index_to_label: dict):
    explainer_config = ExplainerConfig(
        explanation_type="model",
        node_mask_type="object",
        edge_mask_type="object",
    )
    model_config = ModelConfig(
        mode="binary_classification",
        task_level="graph",
        return_type="raw",
    )
    explainer = GNNExplainer(epochs=100)
    explainer.connect(explainer_config, model_config)
    explanation = explainer(
        model=model,
        x=data.x,
        edge_index=data.edge_index,
        target=data.y,
    )
    node_mask = explanation.node_mask
    edge_mask = explanation.edge_mask
    if node_mask.dim() > 1:
        node_mask = node_mask.squeeze(-1)
    if edge_mask.dim() > 1:
        edge_mask = edge_mask.squeeze(-1)

    actionable_nodes: list[dict] = []
    actionable_indices: set[int] = set()

    for idx in range(len(index_to_code)):
        label = index_to_label.get(idx, "")
        score = node_mask[idx].item()
        if label not in ACTIONABLE_TYPES:
            continue
        if score <= 0.25:
            continue
        code_str = (index_to_code.get(idx) or "").strip()
        if not code_str or code_str in {"?", "", "<empty>", "<module>", "<global>"}:
            continue
        if len(code_str) < 2:
            continue
        actionable_nodes.append(
            {
                "index": idx,
                "score": round(float(score), 4),
                "code": code_str,
                "type": label,
            }
        )
        actionable_indices.add(idx)

    actionable_nodes.sort(key=lambda n: n["score"], reverse=True)
    top_nodes = actionable_nodes[:3]

    flows: list[str] = []
    if edge_mask.dim() == 1:
        for e_idx in range(edge_mask.size(0)):
            score = float(edge_mask[e_idx].item())
            if score <= 0.3:
                continue
            if e_idx >= data.edge_index.size(1):
                continue
            src_idx = int(data.edge_index[0, e_idx].item())
            dst_idx = int(data.edge_index[1, e_idx].item())
            if src_idx in actionable_indices and dst_idx in actionable_indices:
                flows.append(
                    f"{index_to_code.get(src_idx, '?')} flows into {index_to_code.get(dst_idx, '?')}"
                )

    return top_nodes, flows[:5]


def _print_result(target: str, num_nodes: int, num_edges: int, probability: float) -> int:
    verdict = "MALICIOUS" if probability >= THRESHOLD else "BENIGN"
    confidence = max(probability, 1.0 - probability) * 100
    print(f"  [!] Analyzing: {target}")
    print(f"  [+] Nodes: {num_nodes} | Edges: {num_edges}")
    print()

    if verdict == "MALICIOUS":
        print(f"  [!!!] VERDICT: {verdict} ({confidence:.1f}% confidence)")
        print()
        print("  [!] Semantic patterns consistent with obfuscated backdoor behavior detected.")
        return 1

    print(f"  [✓]  VERDICT: {verdict} ({confidence:.1f}% confidence)")
    print()
    print("  [✓] No backdoor semantic patterns detected.")
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Semantic CPG Backdoor Detector — v2 semantic inference path",
        epilog="Example:\n  python infer.py --target test_samples/backdoor_sim.py\n",
    )
    parser.add_argument("--target", "-t", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    args = parser.parse_args()

    target = args.target.resolve()
    if not target.is_file():
        print(f"  [ERROR] File not found: {target}", file=__import__("sys").stderr)
        return 2
    if target.suffix != ".py":
        print(f"  [ERROR] Not a .py file: {target}", file=__import__("sys").stderr)
        return 2

    if not VOCAB_PATH.is_file():
        print(f"  [ERROR] Missing vocab: {VOCAB_PATH}", file=__import__("sys").stderr)
        return 2

    if not MODEL_PATH.is_file():
        print(f"  [ERROR] Missing model: {MODEL_PATH}", file=__import__("sys").stderr)
        return 2

    embed_dim = get_embedding_dim()
    model = SemanticBackdoorGNN(embed_dim)
    model.load_state_dict(torch.load(str(MODEL_PATH), weights_only=True))
    model.eval()

    try:
        graph = _load_graph(target)
    except Exception as exc:
        print(f"  [ERROR] CPG extraction failed: {exc}", file=__import__("sys").stderr)
        return 2

    try:
        data, index_to_code, index_to_label = _build_data(graph, embed_dim)
    except Exception as exc:
        print(f"  [ERROR] Graph build failed: {exc}", file=__import__("sys").stderr)
        return 2

    with torch.no_grad():
        logit = model(data)
        probability = torch.sigmoid(logit).item()

    try:
        _explain(data, model, index_to_code, index_to_label)
    except Exception as exc:
        logger.warning("Explanation failed: %s", exc)

    return _print_result(str(target), len(graph["nodes"]), len(graph["edges"]), probability)


if __name__ == "__main__":
    raise SystemExit(main())
