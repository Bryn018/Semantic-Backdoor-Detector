"""
train.py — Train BackdoorDetectorGNN on the PyG dataset.

Loads dataset.pt, splits 50/50 train/test, trains for 50 epochs with
BCEWithLogitsLoss + Adam, prints loss/accuracy every 10 epochs,
and reports final test accuracy.

Usage:
    python train.py [--data PATH] [--epochs N] [--lr FLOAT]
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from gnn_model import BackdoorDetectorGNN

DATA_PATH: Path = Path("dataset.pt")
EPOCHS: int = 50
LR: float = 0.001
TRAIN_SIZE: int = 50
SEED: int = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train")


def train_epoch(model: BackdoorDetectorGNN, loader: DataLoader, optimizer, criterion) -> tuple[float, float]:
    """Train for one epoch.

    Returns:
        (avg_loss, accuracy) over the epoch.
    """
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
def evaluate(model: BackdoorDetectorGNN, loader: DataLoader, criterion) -> tuple[float, float]:
    """Evaluate on a dataset.

    Returns:
        (avg_loss, accuracy).
    """
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
    import argparse
    parser = argparse.ArgumentParser(description="Train BackdoorDetectorGNN.")
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    args = parser.parse_args()

    random.seed(SEED)
    torch.manual_seed(SEED)

    # Load dataset
    logger.info("Loading dataset from %s ...", args.data)
    dataset: list = torch.load(str(args.data), weights_only=False)
    logger.info("Loaded %d graphs", len(dataset))

    # Shuffle and split
    random.shuffle(dataset)
    train_data = dataset[:TRAIN_SIZE]
    test_data = dataset[TRAIN_SIZE:]

    logger.info("Split: %d train, %d test", len(train_data), len(test_data))

    # Determine feature dimension from first graph
    num_features = train_data[0].x.size(1)
    logger.info("Node feature dimension: %d", num_features)

    # Model
    model = BackdoorDetectorGNN(num_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    # DataLoaders (batch_size=10)
    train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=10, shuffle=False)

    print("\n" + "=" * 60)
    print("Training BackdoorDetectorGNN")
    print("=" * 60)
    print(f"  Epochs: {args.epochs} | LR: {args.lr} | Seed: {SEED}")
    print(f"  Train: {len(train_data)} | Test: {len(test_data)}")
    print(f"  Features: {num_features} | Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("=" * 60)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)

        if epoch % 10 == 0 or epoch == 1:
            test_loss, test_acc = evaluate(model, test_loader, criterion)
            print(
                f"  Epoch {epoch:3d}/{args.epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.1f}% | "
                f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.1f}%"
            )

    # Final evaluation
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    train_loss, train_acc = evaluate(model, train_loader, criterion)

    print("\n" + "=" * 60)
    print("Final Results")
    print("=" * 60)
    print(f"  Train Accuracy: {train_acc*100:.1f}%")
    print(f"  Test Accuracy:  {test_acc*100:.1f}%")
    print(f"  Test Loss:      {test_loss:.4f}")
    if test_acc > 0.70:
        print("  >> SUCCESS: GNN learned the backdoor semantic pattern (>70% test acc)")
    else:
        print("  >> Below 70% — may need more data, epochs, or architecture tuning")
    print("=" * 60)


if __name__ == "__main__":
    main()
