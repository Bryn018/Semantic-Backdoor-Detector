#!/usr/bin/env python3
"""
explain.py — Semantic CPG Backdoor Detector with GNNExplainer.

Default path: lightweight CPG -> CodeBERT -> SemanticBackdoorGNN -> GNNExplainer.
Optional Joern fallback is available only when explicitly enabled via CLI.
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import networkx as nx
import os
import sys
import tempfile
import traceback
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
JOERN_DIR: Path = Path(
    os.environ.get("JOERN_DIR", str(Path.home() / "bin" / "joern"))
)
JAVA_HOME = Path(
    os.environ.get("JAVA_HOME", str(Path.home() / ".local" / "jdk" / "jdk-21.0.5+11"))
)

# Actionable node types — code elements a developer can act on
ACTIONABLE_TYPES: set[str] = {
    "CALL",
    "IDENTIFIER",
    "LITERAL",
    "METHOD_PARAMETER_IN",
    "CONTROL_STRUCTURE",
}

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


def _generate_cpg_locally(code_string: str) -> dict:
    """Pure-Python CPG generation using ast + networkx."""
    try:
        tree = ast.parse(code_string)
    except SyntaxError as exc:
        return {
            "nodes": [{"id": 0, "code": f"PARSE_ERROR: {exc}", "label": "UNKNOWN"}],
            "edges": [],
        }

    graph = nx.DiGraph()
    label_map = {
        ast.FunctionDef: "METHOD",
        ast.AsyncFunctionDef: "METHOD",
        ast.ClassDef: "TYPE_DECL",
        ast.Call: "CALL",
        ast.Name: "IDENTIFIER",
        ast.Constant: "LITERAL",
        ast.arg: "METHOD_PARAMETER_IN",
        ast.If: "CONTROL_STRUCTURE",
        ast.For: "CONTROL_STRUCTURE",
        ast.While: "CONTROL_STRUCTURE",
        ast.With: "CONTROL_STRUCTURE",
        ast.Return: "RETURN",
        ast.Assign: "LOCAL",
        ast.AnnAssign: "LOCAL",
        ast.Import: "IMPORT",
        ast.ImportFrom: "IMPORT",
        ast.Module: "MODULE",
        ast.Attribute: "IDENTIFIER",
    }

    def _label(node: ast.AST) -> str:
        for typ, label in label_map.items():  # type: ignore[assignment]
            if isinstance(node, typ):
                return label
        return "UNKNOWN"

    def _code(node: ast.AST) -> str:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return f"def {node.name}"
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute):
                val = getattr(func.value, "id", None) or ast.dump(func.value)
                return f"{val}.{func.attr}"
            if isinstance(func, ast.Name):
                return func.id
            return ast.dump(func)
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Constant):
            return repr(node.value)
        if isinstance(node, ast.Import):
            return "import " + ", ".join(a.name for a in node.names)
        if isinstance(node, ast.ImportFrom):
            return (
                f"from {node.module or ''} import "
                + ", ".join(a.name for a in node.names)
            )
        if isinstance(node, (ast.If, ast.For, ast.While, ast.With, ast.Return)):
            return type(node).__name__.upper()
        if isinstance(node, ast.Assign):
            return ", ".join(t.id for t in node.targets if isinstance(t, ast.Name)) + " ="
        if isinstance(node, ast.Attribute):
            return f"{getattr(node.value, 'id', '?')}.{node.attr}"
        return type(node).__name__

    defs: dict[str, int] = {}

    def add(parent_gid: Optional[int], node: ast.AST) -> int:
        gid = len(graph.nodes)
        label = _label(node)
        code = _code(node)
        graph.add_node(gid, id=gid, code=code, label=label)
        if parent_gid is not None:
            graph.add_edge(parent_gid, gid, label="AST")
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    defs[t.id] = gid
        if isinstance(node, ast.FunctionDef):
            defs[node.name] = gid
        for child in ast.iter_child_nodes(node):
            add(gid, child)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            def_gid = defs.get(node.id)
            if def_gid is not None:
                graph.add_edge(def_gid, gid, label="REACHES")
        return gid

    for top in tree.body:
        add(None, top)

    nodes = [
        {"id": n, "code": d.get("code", ""), "label": d.get("label", "UNKNOWN")}
        for n, d in graph.nodes(data=True)
    ]
    edges = [
        {"src": u, "dst": v, "label": d.get("label", "AST")}
        for u, v, d in graph.edges(data=True)
    ]
    return {"nodes": nodes, "edges": edges}


def extract_cpg(
    py_file: Path,
    output_json: Path,
    use_joern: bool = False,
) -> Optional[dict]:
    """Extract CPG from Python source.

    Args:
        py_file: Python source file.
        output_json: Destination JSON path.
        use_joern: If True, prefer Joern when lightweight CPG yields no nodes.
    """
    try:
        code_string = py_file.read_text(encoding="utf-8")
    except Exception as exc:
        return {"_cpg_error": f"failed to read source: {exc}"}

    graph = _generate_cpg_locally(code_string)
    if graph and graph.get("nodes"):
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(graph, f, indent=2)
        return graph

    if use_joern:
        return _generate_cpg_with_joern(py_file, output_json)

    return {"_cpg_error": "CPG generation produced no nodes"}


def _generate_cpg_with_joern(
    py_file: Path,
    output_json: Path,
) -> Optional[dict]:
    """Optional Joern-based CPG extraction."""
    try:
        from cpg_generator import generate_cpg as generate_cpg_joern
        return generate_cpg_joern(py_file, output_json)
    except Exception as exc:
        logger.error("Joern fallback failed: %s", exc)
        return {"_cpg_error": f"CPG extraction failed: {exc}"}


# ---------------------------------------------------------------------------
# Core Analysis API
# ---------------------------------------------------------------------------


def analyze_code(file_path_or_code: str) -> dict:
    """Analyze a Python file for backdoor patterns with full explainability.

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

    if os.path.exists(file_path_or_code) and file_path_or_code.endswith(".py"):
        py_file = Path(file_path_or_code)
    else:
        temp_py = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
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
            error_msg = (
                graph.get("_cpg_error", "CPG extraction failed")
                if isinstance(graph, dict)
                else "CPG extraction failed"
            )
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
            actionable_nodes.append(
                {
                    "index": idx,
                    "score": round(score, 4),
                    "code": code_str,
                    "type": label,
                }
            )
            actionable_indices.add(idx)

        actionable_nodes.sort(key=lambda n: n["score"], reverse=True)
        top_nodes = actionable_nodes[:3]

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
                        explanation_flows.append(
                            f"{index_to_code.get(src_idx, '?')} flows into {index_to_code.get(dst_idx, '?')}"
                        )

        return {
            "verdict": verdict,
            "confidence": round(confidence, 1),
            "explanation_nodes": [
                {"score": n["score"], "code": n["code"], "type": n["type"]}
                for n in top_nodes
            ],
            "explanation_flows": explanation_flows[:5],
        }

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
    parser.add_argument("--use-joern", action="store_true",
                        help="Allow Joern fallback if lightweight CPG fails.")
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
