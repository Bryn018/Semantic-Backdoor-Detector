#!/usr/bin/env python3
"""
infer.py — Semantic CPG Backdoor Detector CLI.

Analyzes a Python source file for obfuscated backdoor patterns using
a pre-trained Graph Neural Network over Code Property Graphs.

Usage:
    python infer.py --target <path_to_python_file>

Example:
    python infer.py --target test_samples/backdoor_sim.py
    python infer.py --target /path/to/suspicious_module.py

Exit codes:
    0 — BENIGN (no backdoor detected)
    1 — MALICIOUS (backdoor detected)
    2 — Error during analysis
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

from gnn_model import BackdoorDetectorGNN

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH: Path = Path(__file__).parent / "model.pth"
VOCAB_PATH: Path = Path(__file__).parent / "vocab.json"
JOERN_DIR: Path = Path.home() / "bin" / "joern"
JAVA_HOME: Path = Path.home() / ".local" / "jdk" / "jdk-21.0.5+11"
THRESHOLD: float = 0.5  # sigmoid output threshold for malicious/benign

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger("infer")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def joern_env() -> dict[str, str]:
    """Build environment dict with JAVA_HOME and PATH for Joern."""
    env = os.environ.copy()
    env["JAVA_HOME"] = str(JAVA_HOME)
    env["PATH"] = str(JAVA_HOME / "bin") + ":" + env.get("PATH", "")
    return env


def extract_cpg(py_file: Path, output_json: Path) -> Optional[dict]:
    """Extract a Code Property Graph from a Python file using Joern.

    Uses pysrc2cpg to parse the source, then joern-export to produce DOT
    format, and finally consolidates into a single JSON file.

    Args:
        py_file: Absolute path to the Python source file.
        output_json: Path for the consolidated JSON output.

    Returns:
        Parsed graph dict with 'nodes' and 'edges' keys, or None on failure.
    """
    pysrc2cpg = JOERN_DIR / "frontends" / "pysrc2cpg" / "bin" / "pysrc2cpg"
    joern_export = JOERN_DIR / "bin" / "joern-export"

    if not pysrc2cpg.is_file():
        logger.error("pysrc2cpg not found at %s", pysrc2cpg)
        return None

    tmp_cpg = Path(tempfile.gettempdir()) / f"cpg_infer_{os.getpid()}.bin"
    export_dir = Path(tempfile.gettempdir()) / f"dot_infer_{os.getpid()}"

    try:
        if tmp_cpg.is_file():
            tmp_cpg.unlink()
        if export_dir.exists():
            shutil.rmtree(export_dir)

        # Step 1: Parse with pysrc2cpg
        result = subprocess.run(
            [str(pysrc2cpg), str(py_file), "-o", str(tmp_cpg)],
            capture_output=True, text=True, timeout=120,
            env=joern_env(),
        )
        if result.returncode != 0 or not tmp_cpg.is_file():
            logger.error("pysrc2cpg failed: %s", result.stderr[-200:] if result.stderr else "unknown")
            return None

        # Step 2: Export to DOT
        result = subprocess.run(
            [str(joern_export), str(tmp_cpg), "--out", str(export_dir),
             "--repr", "cpg", "--format", "dot"],
            capture_output=True, text=True, timeout=120,
            cwd=str(joern_export.parent), env=joern_env(),
        )
        if result.returncode != 0:
            logger.error("joern-export failed: %s", result.stderr[-200:] if result.stderr else "unknown")
            return None

        # Step 3: Consolidate DOT -> JSON
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
                    "src": int(m.group(1)),
                    "dst": int(m.group(2)),
                    **attrs,
                })

        if not all_nodes:
            logger.error("No nodes extracted from CPG")
            return None

        graph = {"nodes": list(all_nodes.values()), "edges": all_edges}
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(graph, f, indent=2)

        return graph

    except subprocess.TimeoutExpired:
        logger.error("CPG extraction timed out")
        return None
    except OSError as exc:
        logger.error("CPG extraction error: %s", exc)
        return None
    finally:
        if tmp_cpg.is_file():
            tmp_cpg.unlink()
        if export_dir.exists():
            shutil.rmtree(export_dir, ignore_errors=True)


def graph_to_pyg(graph: dict, vocab: dict[str, int]) -> Data:
    """Convert a raw graph dict to a PyG Data object using the global vocabulary.

    Node features are one-hot encoded based on the node type label and
    the global vocabulary. Node types not in the vocabulary are mapped
    to the 'UNKNOWN' index if present, or skipped.

    Args:
        graph: Dict with 'nodes' and 'edges' lists.
        vocab: Global node-type -> index mapping.

    Returns:
        PyG Data object with x, edge_index, and y=0 (dummy label).
    """
    num_types: int = len(vocab)
    unknown_idx: int = vocab.get("UNKNOWN", -1)

    # Map original node IDs to 0-based indices
    id_map: dict[int, int] = {}
    for idx, node in enumerate(graph["nodes"]):
        nid = int(node.get("id", idx))
        id_map[nid] = idx

    # One-hot encode node features
    x = torch.zeros(len(graph["nodes"]), num_types, dtype=torch.float32)
    for idx, node in enumerate(graph["nodes"]):
        label: str = node.get("label", "")
        tidx = vocab.get(label, unknown_idx)
        if tidx >= 0:
            x[idx, tidx] = 1.0

    # Build edge index
    src_list: list[int] = []
    dst_list: list[int] = []
    for edge in graph["edges"]:
        s = int(edge.get("src", -1))
        d = int(edge.get("dst", -1))
        if s in id_map and d in id_map:
            src_list.append(id_map[s])
            dst_list.append(id_map[d])

    if not src_list:
        # No valid edges — add self-loops to avoid empty edge_index
        src_list = list(range(len(graph["nodes"])))
        dst_list = list(range(len(graph["nodes"])))

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    y = torch.tensor([0], dtype=torch.long)  # dummy label

    return Data(x=x, edge_index=edge_index, y=y)


def print_banner() -> None:
    print()
    print("  ╔══════════════════════════════════════════════════════════╗")
    print("  ║       Semantic CPG Backdoor Detector v1.0               ║")
    print("  ║       Graph Neural Network over Code Property Graphs    ║")
    print("  ╚══════════════════════════════════════════════════════════╝")
    print()


def print_result(target: str, num_nodes: int, num_edges: int,
                 probability: float, threshold: float) -> int:
    """Print formatted analysis result. Returns exit code."""
    verdict = "MALICIOUS" if probability >= threshold else "BENIGN"
    confidence = probability if probability >= threshold else (1.0 - probability)
    confidence_pct = confidence * 100

    print(f"  [!] Analyzing: {target}")
    print(f"  [+] Nodes extracted: {num_nodes} | Edges: {num_edges}")
    print()

    if verdict == "MALICIOUS":
        print(f"  [!!!] VERDICT: {verdict} ({confidence_pct:.1f}% confidence)")
        print()
        print(f"  [!] The GNN detected semantic patterns consistent with")
        print(f"      obfuscated backdoor code (base64-encoded command execution,")
        print(f"      reverse shells, data exfiltration, or persistence mechanisms).")
        return 1
    else:
        print(f"  [✓] VERDICT: {verdict} ({confidence_pct:.1f}% confidence)")
        print()
        print(f"  [✓] No backdoor semantic patterns detected.")
        return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Semantic CPG Backdoor Detector — Analyze Python files for obfuscated backdoors using GNNs over Code Property Graphs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n  python infer.py --target test_samples/backdoor_sim.py\n  python infer.py --target /path/to/suspicious_file.py\n",
    )
    parser.add_argument(
        "--target", "-t",
        required=True,
        type=Path,
        help="Path to the Python file to analyze.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=THRESHOLD,
        help=f"Classification threshold (default: {THRESHOLD}).",
    )
    args = parser.parse_args()

    print_banner()

    # Validate target
    target: Path = args.target.resolve()
    if not target.is_file():
        print(f"  [ERROR] File not found: {target}", file=sys.stderr)
        return 2
    if target.suffix != ".py":
        print(f"  [ERROR] Not a Python file: {target}", file=sys.stderr)
        return 2

    # Load vocabulary
    if not VOCAB_PATH.is_file():
        print(f"  [ERROR] Vocabulary not found: {VOCAB_PATH}", file=sys.stderr)
        return 2
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab: dict[str, int] = json.load(f)
    print(f"  [+] Loaded vocabulary: {len(vocab)} node types")

    # Load model
    if not MODEL_PATH.is_file():
        print(f"  [ERROR] Model not found: {MODEL_PATH}", file=sys.stderr)
        return 2
    model: BackdoorDetectorGNN = BackdoorDetectorGNN(len(vocab))
    model.load_state_dict(torch.load(str(MODEL_PATH), weights_only=True))
    model.eval()
    print(f"  [+] Loaded model: {MODEL_PATH.name} ({MODEL_PATH.stat().st_size} bytes)")

    # Extract CPG
    print(f"  [*] Extracting Code Property Graph from {target.name} ...")
    tmp_json = Path(tempfile.gettempdir()) / f"cpg_infer_{os.getpid()}.json"
    try:
        graph = extract_cpg(target, tmp_json)
    finally:
        if tmp_json.is_file():
            tmp_json.unlink()

    if graph is None:
        print(f"  [ERROR] CPG extraction failed for {target.name}", file=sys.stderr)
        return 2

    num_nodes = len(graph["nodes"])
    num_edges = len(graph["edges"])
    print(f"  [+] CPG extracted: {num_nodes} nodes, {num_edges} edges")

    # Convert to PyG
    data = graph_to_pyg(graph, vocab)

    # Inference
    print(f"  [*] Running GNN inference ...")
    with torch.no_grad():
        logit: torch.Tensor = model(data)
        probability: float = torch.sigmoid(logit).item()

    # Print result
    exit_code = print_result(str(target), num_nodes, num_edges, probability, args.threshold)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
