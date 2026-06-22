"""
gnn_model_v2.py — SemanticBackdoorGNN with CodeBERT 768-dim input.

Upgraded from the v1 one-hot architecture (16-dim) to dense semantic
embeddings (768-dim from microsoft/codebert-base). The deeper architecture
with dropout prevents overfitting on the high-dimensional features.

Architecture:
    GCNConv(768 -> 256) -> ReLU -> Dropout(0.3)
    GCNConv(256 -> 64)  -> ReLU -> Dropout(0.3)
    GCNConv(64  -> 16)  -> ReLU -> Dropout(0.3)
    global_mean_pool
    Linear(16 -> 1)
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

logger = logging.getLogger("gnn_model_v2")


class SemanticBackdoorGNN(torch.nn.Module):
    """3-layer GNN for CPG-based backdoor detection with semantic embeddings.

    Takes PyG Data objects with 768-dim node features (CodeBERT embeddings)
    and outputs a single logit for binary classification.

    Args:
        num_node_features: Dimension of input node features (768 for CodeBERT).
    """

    def __init__(self, num_node_features: int = 768) -> None:
        super().__init__()

        self.conv1: GCNConv = GCNConv(num_node_features, 256)
        self.conv2: GCNConv = GCNConv(256, 64)
        self.conv3: GCNConv = GCNConv(64, 16)
        self.dropout: torch.nn.Dropout = torch.nn.Dropout(0.3)
        self.classifier: torch.nn.Linear = torch.nn.Linear(16, 1)

        logger.info(
            "SemanticBackdoorGNN initialized: %d -> 256 -> 64 -> 16 -> 1",
            num_node_features,
        )

    def forward(self, x, edge_index=None, batch=None) -> torch.Tensor:
        """Forward pass through the GNN.

        Supports two calling conventions:
        1. forward(data) — PyG Data object (standard)
        2. forward(x, edge_index) — GNNExplainer compatibility

        Args:
            x: Node feature matrix [N, num_features] or PyG Data object.
            edge_index: Edge COO indices [2, E]. If None, x is treated as Data.
            batch: Batch assignment vector [N]. If None, defaults to all zeros.

        Returns:
            Logit tensor of shape [num_graphs, 1].
        """
        # Handle both calling conventions
        if edge_index is None:
            # Called with Data object
            data = x
            x, edge_index, batch = data.x, data.edge_index, data.batch
        else:
            # Called with (x, edge_index) by GNNExplainer
            if batch is None:
                batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Layer 1: 768 -> 256
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        # Layer 2: 256 -> 64
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        # Layer 3: 64 -> 16
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        # Global mean pooling: [N, 16] -> [batch_size, 16]
        x = global_mean_pool(x, batch)

        # Classifier: [batch_size, 16] -> [batch_size, 1]
        x = self.classifier(x)

        return x


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = SemanticBackdoorGNN(768)
    total = count_parameters(model)
    print(f"\nSemanticBackdoorGNN")
    print(f"  Architecture: 768 -> 256 -> 64 -> 16 -> 1")
    print(f"  Parameters: {total:,}")
    print(f"  Dropout: 0.3 between layers")
