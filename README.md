# Semantic CPG Backdoor Detector

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A Graph Neural Network tool that detects supply chain backdoors in Python packages by learning **semantic data flow patterns** from Code Property Graphs — not just signatures.

Inspired by real-world attacks like **[XZ Utils](https://en.wikipedia.org/wiki/XZ_Utils_backdoor)** (CVE-2024-3094), where a supply chain backdoor hid obfuscated command execution behind layers of base64 encoding and conditional triggers. Traditional signature-based scanners missed it. This tool catches it.

## How it Works

```
Source Code → Code Property Graph (CPG) → 3-Layer GCN → Malicious / Benign
```

1. **Code Property Graph (CPG)** — Joern parses the Python source into a unified graph representation combining AST, control flow, and data flow
2. **Graph Convolutional Network** — A 3-layer GCN (16→64→32→16) learns structural patterns characteristic of backdoor code: base64 decoding → `os.system()`, reverse shell setup, cron persistence, data exfiltration
3. **Verdict** — Sigmoid output gives a probability; ≥0.5 is MALICIOUS

## Architecture

```
┌─────────────────┐     ┌──────────┐     ┌────────────┐     ┌──────────┐
│  Python Source   │────▶│  Joern   │────▶│  PyG Data  │────▶│  GCN     │
│  (.py file)      │     │  CPG     │     │  Object    │     │  Layers  │
└─────────────────┘     └──────────┘     └────────────┘     └────┬─────┘
                                                                  │
                                                     ┌────────────▼────┐
                                                     │  Sigmoid Output  │
                                                     │  P(malicious)    │
                                                     └─────────────────┘
```

## Installation

### Prerequisites
- **Python 3.10+**
- **Java 11+** (required by Joern CPG extractor)
- **Joern CLI** (v4.0+)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/semantic-cpg-backdoor-detector.git
cd semantic-cpg-backdoor-detector

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Install Joern (if not already installed)
curl -L https://github.com/joernio/joern/releases/latest/download/joern-install.sh | bash
```

### Installing Joern Manually
```bash
# Download and extract to ~/bin/joern
mkdir -p ~/bin/joern
cd /tmp
curl -L -o joern-cli.zip https://github.com/joernio/joern/releases/latest/download/joern-cli.zip
unzip joern-cli.zip -d ~/bin/joern
chmod -R +x ~/bin/joern/

# Set environment variables (add to ~/.bashrc)
export JAVA_HOME=$HOME/.local/jdk/jdk-21.0.5+11
export PATH=$JAVA_HOME/bin:$PATH
```

## Usage

```bash
# Analyze a single file
python infer.py --target /path/to/suspicious_file.py

# With custom threshold
python infer.py --target /path/to/file.py --threshold 0.7
```

### Example Output — Malicious File
```
  ╔══════════════════════════════════════════════════════════╗
  ║       Semantic CPG Backdoor Detector v1.0               ║
  ╚══════════════════════════════════════════════════════════╝

  [+] Loaded vocabulary: 16 node types
  [+] Loaded model: model.pth (18097 bytes)
  [*] Extracting Code Property Graph from backdoor.py ...
  [+] CPG extracted: 79 nodes, 231 edges
  [*] Running GNN inference ...

  [!] Analyzing: backdoor.py
  [+] Nodes extracted: 79 | Edges: 231

  [!!!] VERDICT: MALICIOUS (89.7% confidence)

  [!] The GNN detected semantic patterns consistent with
      obfuscated backdoor code (base64-encoded command execution,
      reverse shells, data exfiltration, or persistence mechanisms).
```

### Example Output — Benign File
```
  [✓] VERDICT: BENIGN (92.3% confidence)
```

## Included Model

The repository includes a pre-trained model:

| File | Description |
|------|-------------|
| `model.pth` | Trained GNN weights (3,713 parameters, 18KB) |
| `vocab.json` | Global node-type vocabulary (16 CPG node types) |
| `dataset.pt` | Full training dataset (60 graphs) |

**Out of the box**, you can run `python infer.py --target <file.py>` on any Python file.

## Training

To train on your own dataset:

```bash
# Generate synthetic dataset
python generate_dataset.py

# Re-train the model
python train.py --epochs 50 --lr 0.001
```

## Dataset

The included training set has 60 files:
- **30 benign**: Math utilities, string manipulation, file I/O, data processing
- **30 malicious**: `os.system` + base64, reverse shells, `subprocess.call`, data exfiltration, cron persistence

## Requirements

See [requirements.txt](requirements.txt):
```
torch>=2.0
torch_geometric>=2.0
networkx>=3.0
py4j>=0.10
```

## License

MIT
