#!/usr/bin/env python3
"""Save v2 model artifacts: model_v2.pth and vocab from dataset_semantic.pt."""
import json
import torch
from pathlib import Path
from collections import OrderedDict

from gnn_model_v2 import SemanticBackdoorGNN

# Extract vocab from raw_graphs
raw_dir = Path("raw_graphs")
all_types = set()
for jf in sorted(raw_dir.glob("*.json")):
    with open(jf) as f:
        content = f.read()
    idx = content.find('{')
    if idx < 0: continue
    try:
        graph = json.loads(content[idx:])
    except: continue
    for node in graph.get("nodes", []):
        label = node.get("label")
        if label:
            all_types.add(label)

vocab = OrderedDict(sorted((t, i) for i, t in enumerate(sorted(all_types))))
with open("vocab.json", "w") as f:
    json.dump(vocab, f, indent=2)
print(f"Vocab: {len(vocab)} types")

# Train v2 model quickly and save
from torch_geometric.loader import DataLoader
import random

random.seed(42)
torch.manual_seed(42)

dataset = torch.load("dataset_semantic.pt", weights_only=False)
random.shuffle(dataset)
train_data = dataset[:50]

model = SemanticBackdoorGNN(768)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.BCEWithLogitsLoss()
train_loader = DataLoader(train_data, batch_size=10, shuffle=True)

for epoch in range(1, 51):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y.float().unsqueeze(1))
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in train_loader:
                out = model(batch)
                preds = (torch.sigmoid(out) > 0.5).float()
                correct += (preds == batch.y.float().unsqueeze(1)).sum().item()
                total += batch.num_graphs
        print(f"Epoch {epoch}: acc={100*correct/total:.1f}%")

torch.save(model.state_dict(), "model.pth")
print(f"Saved model.pth ({Path('model.pth').stat().st_size} bytes)")
print("Done.")
