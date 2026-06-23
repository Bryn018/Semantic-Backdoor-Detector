#!/usr/bin/env python3
"""
explain.py — Semantic CPG Backdoor Detector with GNNExplainer.

Analyzes a Python source file for obfuscated backdoor patterns using
CodeBERT + GNN, then explains WHY the model flagged it using
PyTorch Geometric's GNNExplainer.

Usage:
    python explain.py --target <path_to_python_file>

Core API:
    result = analyze_code("suspicious.py")
    # Returns dict with verdict, confidence, explanation_nodes, explanation_flows
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
from torch_geometric.data import Data
from torch_geometric.explain import GNNExplainer, ExplainerConfig, ModelConfig

from gnn_model_v2 import SemanticBackdoorGNN
from embedder import embed_graph_nodes, get_embedding_dim, extract_node_text

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH: Path = Path(__file__).parent / "model.pth"
VOCAB_PATH: Path = Path(__file__).parent / "vocab.json"
JOERN_DIR: Path = Path.home() / "bin" / "joern"
JAVA_HOME = Path.home() / ".local" / "jdk" / "jdk-21.0.5+11"

# Actionable node types — code elements a developer can act on
ACTIONABLE_TYPES: set[str] = {"CALL", "IDENTIFIER", "LITERAL", "METHOD_PARAMETER_IN", "CONTROL_STRUCTURE"}

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger("explain")


# ---------------------------------------------------------------------------
# CPG Extraction
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

    except (subprocess.TimeoutExpired, OSError) as exc:
        import traceback
        error_msg = f"CPG extraction failed:\n{traceback.format_exc()}"
        return {"_cpg_error": error_msg}
    finally:
        if tmp_cpg.is_file():
            tmp_cpg.unlink()
        if export_dir.exists():
            shutil.rmtree(export_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Core Analysis API
# ---------------------------------------------------------------------------

def analyze_code(file_path_or_code: str) -> dict:
    """Analyze a Python file for backdoor patterns with full explainability.

    Extracts a Code Property Graph, embeds nodes with CodeBERT, runs the
    SemanticBackdoorGNN, and uses GNNExplainer to identify the specific
    code elements that drove the verdict.

    Accepts either a file path (CLI) or raw Python code string (Gradio UI).

    Args:
        file_path_or_code: Path to the Python source file, OR raw Python code string.

    Returns:
        A dictionary with the exact structure:
        {
            "verdict": "MALICIOUS" | "BENIGN",
            "confidence": float (0-100),
            "explanation_nodes": [
                {"score": float, "code": str, "type": str},
                ...
            ],
            "explanation_flows": [
                "source_code flows into destination_code",
                ...
            ]
        }
    """
    temp_py_path = None

    # Check if input is a file path or raw code string
    if os.path.exists(file_path_or_code) and file_path_or_code.endswith('.py'):
        py_file = Path(file_path_or_code)
    else:
        # Input is raw code from Gradio UI, save to temp file
        temp_py = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
        temp_py.write(file_path_or_code)
        temp_py.close()
        temp_py_path = temp_py.name
        py_file = Path(temp_py_path)

    try:
        # Load model
        with open(VOCAB_PATH, "r", encoding="utf-8") as f:
            vocab = json.load(f)

        model = SemanticBackdoorGNN(get_embedding_dim())
        model.load_state_dict(torch.load(str(MODEL_PATH), weights_only=True))
        model.eval()

        # Extract CPG
        tmp_json = Path(tempfile.gettempdir()) / f"cpg_analyze_{os.getpid()}.json"
        try:
            graph = extract_cpg(py_file, tmp_json)
        finally:
            if tmp_json.is_file():
                tmp_json.unlink()

        if graph is None or (isinstance(graph, dict) and "_cpg_error" in graph):
            error_msg = graph.get("_cpg_error", "CPG extraction failed") if isinstance(graph, dict) else "CPG extraction failed"
            return {
                "verdict": "ERROR",
                "confidence": 0.0,
                "explanation_nodes": [],
                "explanation_flows": [error_msg],
            }

        # Build PyG Data + mappings
        embed_dim = get_embedding_dim()
        x_np = embed_graph_nodes(graph)
        x = torch.from_numpy(x_np).float()

        # Build mappings
        index_to_code: dict[int, str] = {}
        index_to_label: dict[int, str] = {}
        id_map: dict[int, int] = {}
        for idx, node in enumerate(graph["nodes"]):
            nid = int(node.get("id", idx))
            id_map[nid] = idx
            index_to_code[idx] = extract_node_text(node)
            index_to_label[idx] = node.get("label", "UNKNOWN")

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
        data = Data(x=x, edge_index=edge_index, y=y)

        # Inference
        with torch.no_grad():
            logit = model(data)
            probability = torch.sigmoid(logit).item()

        verdict = "MALICIOUS" if probability >= 0.5 else "BENIGN"
        confidence = max(probability, 1.0 - probability) * 100

        # GNNExplainer
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

        # --- FILTER: Only actionable nodes with score > 0.25 ---
        actionable_nodes: list[dict] = []
        actionable_indices: set[int] = set()

        for idx in range(len(graph["nodes"])):
            label = index_to_label.get(idx, "")
            score = node_mask[idx].item()

            if label not in ACTIONABLE_TYPES:
                continue
            if score <= 0.25:
                continue

            code_str = index_to_code.get(idx, "")
            if not code_str or code_str in ("?", "", "<empty>", "<module>", "<global>"):
                continue
            if len(code_str.strip()) < 2:
                continue

            actionable_nodes.append({
                "index": idx,
                "score": round(score, 4),
                "code": code_str,
                "type": label,
            })
            actionable_indices.add(idx)

        actionable_nodes.sort(key=lambda n: n["score"], reverse=True)
        top_nodes = actionable_nodes[:3]

        # --- FILTER: Only edges where BOTH src and dst are actionable ---
        explanation_flows: list[str] = []
        if edge_mask.dim() == 1:
            for e_idx in range(edge_mask.size(0)):
                score = edge_mask[e_idx].item()
                if score <= 0.3:
                    continue
                if e_idx < data.edge_index.size(1):
                    src_idx = data.edge_index[0, e_idx].item()
                    dst_idx = data.edge_index[1, e_idx].item()
                    if src_idx in actionable_indices and dst_idx in actionable_indices:
                        src_code = index_to_code.get(src_idx, "?")
                        dst_code = index_to_code.get(dst_idx, "?")
                        explanation_flows.append(
                            f"{src_code} flows into {dst_code}"
                        )

        result: dict = {
            "verdict": verdict,
            "confidence": round(confidence, 1),
            "explanation_nodes": [
                {"score": n["score"], "code": n["code"], "type": n["type"]}
                for n in top_nodes
            ],
            "explanation_flows": explanation_flows[:5],
        }

        return result

    finally:
        if temp_py_path and os.path.exists(temp_py_path):
            os.remove(temp_py_path)



def print_banner() -> None:
    print()
    print("  ╔══════════════════════════════════════════════════════════════╗")
    print("  ║     Semantic CPG Backdoor Detector v2.1 — Explainable AI    ║")
    print("  ║     CodeBERT + GCN + GNNExplainer                           ║")
    print("  ╚══════════════════════════════════════════════════════════════╝")
    print()


def print_result(result: dict, target: str) -> int:
    """Print formatted result from analyze_code() dict. Returns exit code."""
    verdict = result.get("verdict", "ERROR")
    confidence = result.get("confidence", 0.0)

    print(f"  [!] Analyzing: {target}")
    print()

    if verdict == "ERROR":
        print(f"  [ERROR] Analysis failed")
        for flow in result.get("explanation_flows", []):
            print(f"    -> {flow}")
        return 2

    if verdict == "MALICIOUS":
        print(f"  [!!!] VERDICT: {verdict} ({confidence:.1f}% confidence)")
    else:
        print(f"  [✓]  VERDICT: {verdict} ({confidence:.1f}% confidence)")

    nodes = result.get("explanation_nodes", [])
    flows = result.get("explanation_flows", [])

    if nodes or flows:
        print()
        print("  ─────────────────────────────────────────")
        print("  --- EXPLANATION (GNNExplainer) ---")
        print("  ─────────────────────────────────────────")
        print()

        if nodes:
            print("  Critical Code Elements:")
            for rank, node in enumerate(nodes, 1):
                score = node["score"]
                code = node["code"]
                ntype = node["type"]
                if len(code) > 65:
                    code = code[:62] + "..."
                print(f"  {rank}. [Score: {score:.2f}] [{ntype}] \"{code}\"")

        if flows:
            print()
            print("  Critical Data Flows:")
            for flow in flows[:5]:
                print(f"  -> {flow}")

    print()
    return 1 if verdict == "MALICIOUS" else 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Semantic CPG Backdoor Detector with GNNExplainer explanations.",
    )
    parser.add_argument("--target", "-t", required=True, type=str,
                        help="Path to the Python file to analyze.")
    parser.add_argument("--json", action="store_true",
                        help="Output raw JSON dictionary.")
    args = parser.parse_args()

    if not args.json:
        print_banner()

    result = analyze_code(args.target)

    if args.json:
        print(json.dumps(result, indent=2))
        exit_code = 1 if result.get("verdict") == "MALICIOUS" else 0
    else:
        exit_code = print_result(result, args.target)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
