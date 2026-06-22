"""
train_v2.py — Train SemanticBackdoorGNN on 768-dim CodeBERT embeddings.

Loads dataset_semantic.pt, splits 50/50 train/test, trains for 50 epochs
with BCEWithLogitsLoss + Adam. Prints loss and accuracy every 10 epochs.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from gnn_model_v2 import SemanticBackdoorGNN, count_parameters

DATA_PATH: Path = Path("dataset_semantic.pt")
EPOCHS: int = 50
LR: float = 0.001
BATCH_SIZE: int = 10
TRAIN_SIZE: int = 50
SEED: int = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_v2")


def train_epoch(model, loader, optimizer, criterion) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch in loader:
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y.float().unsqueeze(1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        preds = (torch.sigmoid(out) > 0.5).float()
        correct += (preds == batch.y.float().unsqueeze(1)).sum().item()
        total += batch.num_graphs
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch in loader:
        out = model(batch)
        loss = criterion(out, batch.y.float().unsqueeze(1))
        total_loss += loss.item() * batch.num_graphs
        preds = (torch.sigmoid(out) > 0.5).float()
        correct += (preds == batch.y.float().unsqueeze(1)).sum().item()
        total += batch.num_graphs
    return total_loss / total, correct / total


def main() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)

    logger.info("Loading dataset from %s ...", DATA_PATH)
    dataset = torch.load(str(DATA_PATH), weights_only=False)
    logger.info("Loaded %d graphs", len(dataset))

    # Shuffle and split
    random.shuffle(dataset)
    train_data = dataset[:TRAIN_SIZE]
    test_data = dataset[TRAIN_SIZE:]

    logger.info("Split: %d train, %d test", len(train_data), len(test_data))

    # Feature dimension from first graph
    num_features = train_data[0].x.size(1)
    logger.info("Node feature dimension: %d", num_features)

    # Model
    model = SemanticBackdoorGNN(num_features)
    total_params = count_parameters(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.BCEWithLogitsLoss()

    # DataLoaders
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    print("\n" + "=" * 60)
    print("Training SemanticBackdoorGNN (CodeBERT 768-dim)")
    print("=" * 60)
    print(f"  Epochs: {EPOCHS} | LR: {LR} | Seed: {SEED}")
    print(f"  Train: {len(train_data)} | Test: {len(test_data)}")
    print(f"  Features: {num_features} | Parameters: {total_params:,}")
    print("=" * 60)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)

        if epoch % 10 == 0 or epoch == 1:
            test_loss, test_acc = evaluate(model, test_loader, criterion)
            print(
                f"  Epoch {epoch:3d}/{EPOCHS} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.1f}% | "
                f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc * 100:.1f}%"
            )

    # Final evaluation
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    train_loss, train_acc = evaluate(model, train_loader, criterion)

    print("\n" + "=" * 60)
    print("Final Results — SemanticBackdoorGNN v2")
    print("=" * 60)
    print(f"  Train Accuracy: {train_acc * 100:.1f}%")
    print(f"  Test Accuracy:  {test_acc * 100:.1f}%")
    print(f"  Test Loss:      {test_loss:.4f}")
    if test_acc > 0.70:
        print("  >> SUCCESS: GNN learned semantic backdoor patterns from CodeBERT embeddings")
    else:
        print("  >> Below 70% — may need more data or architecture tuning")
    print("=" * 60)


if __name__ == "__main__":
    main()
