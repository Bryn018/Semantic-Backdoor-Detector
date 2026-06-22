#!/usr/bin/env python3
"""
explain.py — Semantic CPG Backdoor Detector with GNNExplainer.

Analyzes a Python source file for obfuscated backdoor patterns using
CodeBERT + GNN, then explains WHY the model flagged it using
PyTorch Geometric's GNNExplainer to identify critical nodes and edges.

Usage:
    python explain.py --target <path_to_python_file>
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.explain import GNNExplainer

from gnn_model_v2 import SemanticBackdoorGNN
from embedder import embed_graph_nodes, get_embedding_dim, extract_node_text

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH: Path = Path(__file__).parent / "model.pth"
VOCAB_PATH: Path = Path(__file__).parent / "vocab.json"
JOERN_DIR: Path = Path.home() / "bin" / "joern"
JAVA_HOME = Path.home() / ".local" / "jdk" / "jdk-21.0.5+11"

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger("explain")


# ---------------------------------------------------------------------------
# CPG Extraction (same pipeline as infer.py)
# ---------------------------------------------------------------------------


def joern_env() -> dict[str, str]:
    env = os.environ.copy()
    env["JAVA_HOME"] = str(JAVA_HOME)
    env["PATH"] = str(JAVA_HOME / "bin") + ":" + env.get("PATH", "")
    return env


def extract_cpg(py_file: Path, output_json: Path) -> Optional[dict]:
    """Extract CPG from Python file using pysrc2cpg + joern-export."""
    pysrc2cpg = JOERN_DIR / "frontends" / "pysrc2cpg" / "bin" / "pysrc2cpg"
    joern_export = JOERN_DIR / "bin" / "joern-export"

    if not pysrc2cpg.is_file():
        logger.error("pysrc2cpg not found at %s", pysrc2cpg)
        return None

    tmp_cpg = Path(tempfile.gettempdir()) / f"cpg_explain_{os.getpid()}.bin"
    export_dir = Path(tempfile.gettempdir()) / f"dot_explain_{os.getpid()}"

    try:
        if tmp_cpg.is_file():
            tmp_cpg.unlink()
        if export_dir.exists():
            shutil.rmtree(export_dir)

        result = subprocess.run(
            [str(pysrc2cpg), str(py_file), "-o", str(tmp_cpg)],
            capture_output=True, text=True, timeout=120, env=joern_env(),
        )
        if result.returncode != 0 or not tmp_cpg.is_file():
            logger.error("pysrc2cpg failed")
            return None

        result = subprocess.run(
            [str(joern_export), str(tmp_cpg), "--out", str(export_dir),
             "--repr", "cpg", "--format", "dot"],
            capture_output=True, text=True, timeout=120,
            cwd=str(joern_export.parent), env=joern_env(),
        )
        if result.returncode != 0:
            logger.error("joern-export failed")
            return None

        all_nodes: dict[int, dict] = {}
        all_edges: list[dict] = []
        node_re = re.compile(r'"(\d+)"\s*\[([^\]]+)\]')
        edge_re = re.compile(r'"(\d+)"\s*->\s*"(\d+)"\s*\[([^\]]+)\]')
        attr_re = re.compile(r'(\w+)="([^"]*)"')

        for dot_file in export_dir.rglob("export.dot"):
            content = dot_file.read_text(encoding="utf-8")
            for m in node_re.finditer(content):
                nid = int(m.group(1))
                attrs = dict(attr_re.findall(m.group(2)))
                if nid not in all_nodes:
                    all_nodes[nid] = {"id": nid, **attrs}
            for m in edge_re.finditer(content):
                attrs = dict(attr_re.findall(m.group(3)))
                all_edges.append({
                    "src": int(m.group(1)), "dst": int(m.group(2)), **attrs,
                })

        if not all_nodes:
            return None

        graph = {"nodes": list(all_nodes.values()), "edges": all_edges}
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(graph, f, indent=2)
        return graph

    except (subprocess.TimeoutExpired, OSError):
        return None
    finally:
        if tmp_cpg.is_file():
            tmp_cpg.unlink()
        if export_dir.exists():
            shutil.rmtree(export_dir, ignore_errors=True)


def graph_to_pyg_with_mapping(graph: dict) -> tuple[Data, dict[int, str], dict[int, int]]:
    """Convert graph to PyG Data + build node_index -> code_text mapping.

    Returns:
        (Data object, dict mapping node index to original code string)
    """
    embed_dim = get_embedding_dim()
    x_np = embed_graph_nodes(graph)
    x = torch.from_numpy(x_np).float()

    # Build mapping: node_index -> original code string
    index_to_code: dict[int, str] = {}
    id_map: dict[int, int] = {}
    for idx, node in enumerate(graph["nodes"]):
        nid = int(node.get("id", idx))
        id_map[nid] = idx
        index_to_code[idx] = extract_node_text(node)

    src_list, dst_list = [], []
    for edge in graph["edges"]:
        s = int(edge.get("src", -1))
        d = int(edge.get("dst", -1))
        if s in id_map and d in id_map:
            src_list.append(id_map[s])
            dst_list.append(id_map[d])

    if not src_list:
        src_list = list(range(len(graph["nodes"])))
        dst_list = list(range(len(graph["nodes"])))

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    y = torch.tensor([0], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y), index_to_code, id_map


# ---------------------------------------------------------------------------
# Formatted Output
# ---------------------------------------------------------------------------


def print_banner() -> None:
    print()
    print("  ╔══════════════════════════════════════════════════════════════╗")
    print("  ║     Semantic CPG Backdoor Detector v2.0 — With Explanations ║")
    print("  ║     CodeBERT + GCN + GNNExplainer                           ║")
    print("  ╚══════════════════════════════════════════════════════════════╝")
    print()


def print_explanation(
    target: str,
    num_nodes: int,
    num_edges: int,
    probability: float,
    node_mask: torch.Tensor,
    edge_mask: torch.Tensor,
    index_to_code: dict[int, str],
    id_map: dict[int, int],
    data: Data,
) -> int:
    """Print formatted analysis + explanation."""
    verdict = "MALICIOUS" if probability >= 0.5 else "BENIGN"
    confidence = max(probability, 1.0 - probability) * 100

    print(f"  [!] Analyzing: {target}")
    print(f"  [+] Nodes extracted: {num_nodes} | Edges: {num_edges}")
    print()

    if verdict == "MALICIOUS":
        print(f"  [!!!] VERDICT: {verdict} ({confidence:.1f}% confidence)")
    else:
        print(f"  [✓]  VERDICT: BENIGN ({confidence:.1f}% confidence)")

    # --- EXPLANATION ---
    print()
    print("  ─────────────────────────────────────────")
    print("  --- EXPLANATION (GNNExplainer) ---")
    print("  ─────────────────────────────────────────")
    print()

    # Top critical nodes — filter out pure structural nodes
    STRUCTURAL_LABELS = {"METHOD", "METHOD_RETURN", "MODIFIER", "TYPE_DECL",
                         "BLOCK", "RETURN", "IMPORT"}
    node_scores = [(i, node_mask[i].item()) for i in range(len(index_to_code))]
    node_scores.sort(key=lambda x: x[1], reverse=True)

    # Prefer nodes with actual code text, skip pure structural
    meaningful_nodes = []
    for idx, score in node_scores:
        code_str = index_to_code.get(idx, "?")
        # Skip nodes with empty or purely structural text
        if not code_str or code_str in ("?", "", "<empty>", "<module>", "<global>"):
            continue
        if len(code_str) < 2:
            continue
        meaningful_nodes.append((idx, score, code_str))
        if len(meaningful_nodes) >= 5:
            break

    print("  Critical Nodes Detected:")
    if meaningful_nodes:
        for rank, (idx, score, code_str) in enumerate(meaningful_nodes[:3], 1):
            display = code_str if len(code_str) <= 60 else code_str[:57] + "..."
            print(f"  {rank}. [Score: {score:.2f}] \"{display}\"")
    else:
        # Fallback to top 3 raw if nothing meaningful found
        for rank, (idx, score) in enumerate(node_scores[:3], 1):
            code_str = index_to_code.get(idx, "?")
            display = code_str if len(code_str) <= 60 else code_str[:57] + "..."
            print(f"  {rank}. [Score: {score:.2f}] \"{display}\"")

    print()

    # Top critical data flow edges
    edge_scores = []
    if edge_mask.dim() == 1:
        for e_idx in range(edge_mask.size(0)):
            edge_scores.append((e_idx, edge_mask[e_idx].item()))
    edge_scores.sort(key=lambda x: x[1], reverse=True)

    if edge_scores:
        print("  Critical Data Flow Detected:")
        shown = 0
        for e_idx, score in edge_scores:
            if shown >= 2:
                break
            # Map edge index back to src/dst node text
            # edge_index[:, e_idx] gives us [src, dst]
            if e_idx < data.edge_index.size(1):
                src_idx = data.edge_index[0, e_idx].item()
                dst_idx = data.edge_index[1, e_idx].item()
                src_text = index_to_code.get(src_idx, "?")
                dst_text = index_to_code.get(dst_idx, "?")
                if len(src_text) > 40:
                    src_text = src_text[:37] + "..."
                if len(dst_text) > 40:
                    dst_text = dst_text[:37] + "..."
                print(f"  -> [{score:.2f}] \"{src_text}\" => \"{dst_text}\"")
                shown += 1
        if shown == 0:
            for e_idx, score in edge_scores[:2]:
                print(f"  -> Edge #{e_idx} [Score: {score:.2f}] contributes to malicious pattern")

    print()
    return 1 if verdict == "MALICIOUS" else 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Semantic CPG Backdoor Detector with GNNExplainer explanations.",
    )
    parser.add_argument("--target", "-t", required=True, type=Path,
                        help="Path to the Python file to analyze.")
    args = parser.parse_args()

    print_banner()

    target: Path = args.target.resolve()
    if not target.is_file():
        print(f"  [ERROR] File not found: {target}", file=sys.stderr)
        return 2

    # Load vocab
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab: dict[str, int] = json.load(f)

    # Load model
    model = SemanticBackdoorGNN(get_embedding_dim())
    model.load_state_dict(torch.load(str(MODEL_PATH), weights_only=True))
    model.eval()

    # Extract CPG
    tmp_json = Path(tempfile.gettempdir()) / f"cpg_explain_{os.getpid()}.json"
    try:
        graph = extract_cpg(target, tmp_json)
    finally:
        if tmp_json.is_file():
            tmp_json.unlink()

    if graph is None:
        print(f"  [ERROR] CPG extraction failed", file=sys.stderr)
        return 2

    # Convert to PyG
    data, index_to_code, id_map = graph_to_pyg_with_mapping(graph)
    num_nodes = data.x.size(0)
    num_edges = data.edge_index.size(1)

    # Inference
    with torch.no_grad():
        logit = model(data)
        probability = torch.sigmoid(logit).item()

    # GNNExplainer
    print(f"  [*] Running GNNExplainer (this may take a moment) ...")
    from torch_geometric.explain import ExplainerConfig, ModelConfig
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

    # Print results + explanation
    exit_code = print_explanation(
        str(target), num_nodes, num_edges, probability,
        node_mask, edge_mask, index_to_code, id_map, data,
    )
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
