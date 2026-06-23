# Semantic CPG Backdoor Detector

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Release](https://img.shields.io/github/v/release/Bryn018/Semantic-Backdoor-Detector)

A Graph Neural Network tool that detects supply chain backdoors in Python packages by learning **semantic data flow patterns** from Code Property Graphs — not just signatures.

Inspired by real-world attacks like **[XZ Utils](https://en.wikipedia.org/wiki/XZ_Utils_backdoor)** (CVE-2024-3094), where a supply chain backdoor hid obfuscated command execution behind layers of base64 encoding and conditional triggers. Traditional signature-based scanners missed it. This tool catches it.

## How it Works

```
Source Code → Code Property Graph (CPG) → CodeBERT Embeddings → GNN → Malicious / Benign
```

1. **Code Property Graph (CPG)** — Python source is parsed into a graph combining AST, control flow, and data flow
2. **CodeBERT Embeddings** — Each CPG node is embedded into a 768-dim vector using `microsoft/codebert-base`
3. **Graph Neural Network** — A 3-layer GCN (768 → 256 → 64 → 1) learns structural patterns characteristic of backdoor code: base64 decoding → `os.system()`, reverse shell setup, cron persistence, data exfiltration
4. **Verdict** — Sigmoid output gives a probability; ≥0.5 is MALICIOUS
5. **GNNExplainer** — Highlights the specific nodes and data flows that drove the verdict

## Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌──────────────────┐     ┌──────────┐
│  Python Source   │────▶│  lightweight │────▶│  CodeBERT + PyG  │────▶│  GNN     │
│  (.py file)      │     │  CPG (pure   │     │  Data (768-dim)  │     │  Layers  │
└─────────────────┘     │  Python)    │     └──────────────────┘     └────┬─────┘
                         └──────────────┘                                       │
                                                                   ┌────────────▼────┐
                                                                   │  Sigmoid Output  │
                                                                   │  P(malicious)    │
                                                                   └─────────────────┘
```

## Installation

### Prerequisites
- **Python 3.10+**
- No Java / Joern required for default usage (optional Joern fallback still supported)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/Bryn018/Semantic-Backdoor-Detector.git
cd Semantic-Backdoor-Detector

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

## Usage

### Command Line

```bash
# Analyze a single file
python explain.py --target /path/to/suspicious_file.py

# JSON output for automation / pipelines
python explain.py --target /path/to/file.py --json
```

### Web UI (Gradio)

```bash
python app.py
```

Then open `http://localhost:7860` and paste Python source into the textbox.

### Example Output — Malicious File

```json
{
  "verdict": "MALICIOUS",
  "confidence": 87.6,
  "explanation_nodes": [
    {"score": 0.8421, "code": "_decode_payload", "type": "METHOD"},
    {"score": 0.7214, "code": "base64.b64decode", "type": "CALL"}
  ],
  "explanation_flows": [
    "encoded_cmd flows into base64.b64decode",
    "decoded_result flows into os.system"
  ]
}
```

### Example Output — Benign File

```json
{
  "verdict": "BENIGN",
  "confidence": 99.7,
  "explanation_nodes": [],
  "explanation_flows": []
}
```

## Included Model

The repository includes a pre-trained model:

| File | Description |
|------|-------------|
| `model.pth` | Trained GNN weights (214,369 parameters, ~860KB) |
| `vocab.json` | Global node-type vocabulary (15 CPG node types) |
| `embedding_cache.json` | Cached CodeBERT embeddings (auto-generated) |

**Out of the box**, you can run `python explain.py --target <file.py>` on any Python file.

## Training

To train on your own dataset:

```bash
# Generate synthetic dataset
python generate_dataset.py

# Re-train the model
python train_v2.py
```

## Dataset

The included training set has 60 files:
- **30 benign**: Math utilities, string manipulation, file I/O, data processing
- **30 malicious**: `os.system` + base64, reverse shells, `subprocess.call`, data exfiltration, cron persistence

## CPG Backends

| Backend | Environment | Notes |
|---------|-------------|-------|
| `lightweight` (default) | Pure Python | Uses `ast` + `networkx`. No JVM required. Fast. |
| `joern` | Optional fallback | Set `CPG_BACKEND=joern`. Requires Java + Joern CLI. |

```bash
# Force Joern backend
CPG_BACKEND=joern python explain.py --target file.py
```

## CI/CD

This repository ships a GitHub Actions workflow that automatically deploys `app.py`, `Dockerfile`, `explain.py`, `gnn_model_v2.py`, `embedder.py`, `model.pth`, `vocab.json`, and `requirements.txt` to the Hugging Face Space on every push to `master`.

Required secret: `HF_TOKEN` (user access token with `write` scope on the target Space).

## Hugging Face Space

Live demo: [swertay/semantic-backdoor-detector](https://huggingface.co/spaces/swertay/semantic-backdoor-detector)

## Requirements

See [requirements.txt](requirements.txt):
```
torch>=2.0
torch_geometric>=2.0
networkx>=3.0
transformers>=4.0
gradio>=4.0
```

## License

This project is licensed under the [MIT License](LICENSE).

**Responsible Use Notice:** This tool is intended for defensive security research — detecting supply chain backdoors in dependencies, auditing your own codebases, and authorized red-team/blue-team exercises. It is NOT intended for evading detection of malicious code, unauthorized scanning, or any activity violating applicable laws. See [LICENSE](LICENSE) for the full terms and responsible use notice.
