"""
build_pyg_dataset_v2.py — Build PyG dataset with 768-dim CodeBERT embeddings.

Replaces the one-hot node type encoding (16-dim) with dense semantic
embeddings from CodeBERT (768-dim). Each node's CODE/NAME text is
passed through microsoft/codebert-base to produce a meaningful vector
that captures what the code actually does.

Output: dataset_semantic.pt — list of PyG Data objects with x.shape = [N, 768]
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

from embedder import embed_graph_nodes, get_embedding_dim, extract_node_text

RAW_GRAPHS_DIR: Path = Path("raw_graphs")
OUTPUT_PATH: Path = Path("dataset_semantic.pt")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("build_v2")


def load_raw_graph(json_path: Path) -> dict | None:
    """Load a raw CPG JSON file, handling any leading non-JSON text."""
    with open(json_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Find the first '{' that starts the JSON object
    idx = content.find('{')
    if idx < 0:
        logger.warning("No JSON object found in %s", json_path.name)
        return None

    try:
        return json.loads(content[idx:])
    except json.JSONDecodeError as exc:
        logger.warning("JSON parse error in %s: %s", json_path.name, exc)
        return None


def build_dataset() -> list[Data]:
    """Build the full semantic dataset from raw CPG JSON files."""
    json_files = sorted(RAW_GRAPHS_DIR.glob("*.json"))
    dataset: list[Data] = []
    embed_dim: int = get_embedding_dim()

    logger.info("Building semantic dataset from %d graphs ...", len(json_files))
    logger.info("Embedding dimension: %d", embed_dim)

    for jf in json_files:
        graph = load_raw_graph(jf)
        if graph is None:
            continue

        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])

        if not nodes:
            logger.warning("Skipping %s: no nodes", jf.name)
            continue

        # Extract 768-dim CodeBERT embeddings for each node
        try:
            x_np: np.ndarray = embed_graph_nodes(graph)
        except Exception as exc:
            logger.error("Embedding failed for %s: %s", jf.name, exc)
            continue

        x = torch.from_numpy(x_np).float()

        # Map original node IDs to 0-based indices
        id_map: dict[int, int] = {}
        for idx, node in enumerate(nodes):
            nid = int(node.get("id", idx))
            id_map[nid] = idx

        # Build edge index
        src_list: list[int] = []
        dst_list: list[int] = []
        for edge in edges:
            s = int(edge.get("src", -1))
            d = int(edge.get("dst", -1))
            if s in id_map and d in id_map:
                src_list.append(id_map[s])
                dst_list.append(id_map[d])

        if not src_list:
            src_list = list(range(len(nodes)))
            dst_list = list(range(len(nodes)))

        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

        # Label from filename prefix
        label = 1 if jf.stem.startswith("malicious") else 0
        y = torch.tensor([label], dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, y=y)
        data.name = jf.stem
        dataset.append(data)

        logger.info(
            "%s: %d nodes, %d edges, label=%d, x=%s",
            jf.name, len(nodes), len(edges), label, tuple(x.shape),
        )

    return dataset


def main() -> None:
    logger.info("=" * 60)
    logger.info("Semantic Dataset Builder v2 — CodeBERT 768-dim embeddings")
    logger.info("=" * 60)

    dataset = build_dataset()

    if not dataset:
        logger.error("No valid graphs found!")
        return

    torch.save(dataset, str(OUTPUT_PATH))

    benign = sum(1 for d in dataset if d.y.item() == 0)
    malicious = sum(1 for d in dataset if d.y.item() == 1)

    print(f"\nDataset: {len(dataset)} graphs ({benign} benign, {malicious} malicious)")
    print(f"Feature dim: {get_embedding_dim()}")
    print(f"Saved to: {OUTPUT_PATH}")
    print(f"File size: {OUTPUT_PATH.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
