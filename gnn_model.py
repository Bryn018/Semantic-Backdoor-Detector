"""
gnn_model.py — BackdoorDetectorGNN model definition.

Defines a 3-layer Graph Convolutional Network (GCN) for binary
classification of Code Property Graphs (malicious vs. benign).

Architecture:
    GCNConv(num_node_features -> 64) -> ReLU
    GCNConv(64 -> 32)                 -> ReLU
    GCNConv(32 -> 16)                 -> ReLU
    global_mean_pool
    Linear(16 -> 1)                   -> logit output

Usage:
    python gnn_model.py [--data PATH]
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_DATA_PATH: Path = Path(__file__).parent / "cpg_pytorch.pt"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger: logging.Logger = logging.getLogger("gnn_model")


class BackdoorDetectorGNN(torch.nn.Module):
    """3-layer GNN for CPG-based backdoor detection.

    Takes a PyG Data object with node features and edge indices,
    passes through 3 GCN layers with ReLU activations, applies
    global mean pooling to produce a graph-level embedding, and
    outputs a single logit for binary classification.

    Args:
        num_node_features: Dimension of input node feature vectors
            (i.e., number of unique node types in the one-hot encoding).
    """

    def __init__(self, num_node_features: int) -> None:
        super().__init__()

        self.conv1: GCNConv = GCNConv(num_node_features, 64)
        self.conv2: GCNConv = GCNConv(64, 32)
        self.conv3: GCNConv = GCNConv(32, 16)
        self.classifier: torch.nn.Linear = torch.nn.Linear(16, 1)

        logger.info(
            "BackdoorDetectorGNN initialized: %d -> 64 -> 32 -> 16 -> 1",
            num_node_features,
        )

    def forward(
        self,
        data: "torch_geometric.data.Data",
    ) -> torch.Tensor:
        """Forward pass through the GCN.

        Args:
            data: PyG Data object with attributes:
                - x: Node feature matrix [N, num_node_features]
                - edge_index: Edge COO indices [2, E]
                - batch: Batch assignment vector [N] (optional;
                  defaults to single-graph batch of all zeros)

        Returns:
            Logit tensor of shape [1] for binary classification.
        """
        x: torch.Tensor = data.x
        edge_index: torch.Tensor = data.edge_index
        batch: torch.Tensor = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Layer 1: num_node_features -> 64
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        # Layer 2: 64 -> 32
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Layer 3: 32 -> 16
        x = self.conv3(x, edge_index)
        x = F.relu(x)

        # Global mean pooling: [N, 16] -> [1, 16]
        x = global_mean_pool(x, batch)

        # Classifier: [1, 16] -> [1, 1]
        x = self.classifier(x)

        return x


def load_num_features(data_path: Path) -> int:
    """Load the saved PyG Data and return num_node_features.

    Args:
        data_path: Path to the .pt file.

    Returns:
        Number of node features (columns in data.x).

    Raises:
        FileNotFoundError: If the data file does not exist.
    """
    if not data_path.is_file():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    data = torch.load(str(data_path), weights_only=False)
    num_features: int = data.x.size(1)
    logger.info(
        "Loaded data from %s: x=%s, num_features=%d",
        data_path,
        tuple(data.x.shape),
        num_features,
    )
    return num_features


def main() -> None:
    """CLI entry point — instantiate and summarize the model."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Define and summarize the BackdoorDetectorGNN."
    )
    parser.add_argument(
        "--data", "-d",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"Path to cpg_pytorch.pt (default: {DEFAULT_DATA_PATH})",
    )
    args = parser.parse_args()

    try:
        num_features: int = load_num_features(args.data)
        model: BackdoorDetectorGNN = BackdoorDetectorGNN(num_features)

        print("\n" + "=" * 60)
        print("BackdoorDetectorGNN Architecture")
        print("=" * 60)
        print(model)
        print("-" * 60)

        # Parameter count
        total_params: int = sum(p.numel() for p in model.parameters())
        trainable_params: int = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        print(f"Total parameters:     {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print("=" * 60)

    except FileNotFoundError as exc:
        logger.error("Failed to load data: %s", exc)
        print(f"\nERROR: {exc}", file=__import__("sys").stderr)
        __import__("sys").exit(1)


if __name__ == "__main__":
    main()
