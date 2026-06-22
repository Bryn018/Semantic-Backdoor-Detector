"""
embedder.py — CodeBERT dense embedding engine for CPG node features.

Loads microsoft/codebert-base from HuggingFace and provides a cached
mean-pooled embedding function. Each node's CODE or NAME attribute
is encoded into a 768-dimensional dense vector that captures semantic
meaning of the code text.

Usage:
    from embedder import get_code_embedding, get_embedding_dim
    vec = get_code_embedding("os.system")  # returns 768-dim list
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME: str = "microsoft/codebert-base"
EMBEDDING_DIM: int = 768
CACHE_PATH: Path = Path("embedding_cache.json")
MAX_SEQ_LENGTH: int = 128

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(name)s: %(message)s")
logger = logging.getLogger("embedder")

# ---------------------------------------------------------------------------
# Model & Tokenizer (loaded once, lazily)
# ---------------------------------------------------------------------------

_tokenizer: Optional[object] = None
_model: Optional[object] = None
_device: torch.device = torch.device("cpu")
_embedding_cache: dict[str, list] = {}


def _load_model() -> None:
    """Lazy-load CodeBERT model and tokenizer."""
    global _tokenizer, _model

    if _tokenizer is not None and _model is not None:
        return

    from transformers import AutoTokenizer, AutoModel

    logger.info("Loading CodeBERT model: %s ...", MODEL_NAME)
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    _model = AutoModel.from_pretrained(MODEL_NAME)
    _model.to(_device)
    _model.eval()
    logger.info("CodeBERT loaded. Embedding dim: %d", EMBEDDING_DIM)


def _load_cache() -> None:
    """Load embedding cache from disk."""
    global _embedding_cache

    if _embedding_cache:
        return

    if CACHE_PATH.is_file():
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            _embedding_cache = json.load(f)
        logger.info("Loaded %d cached embeddings from %s", len(_embedding_cache), CACHE_PATH)
    else:
        _embedding_cache = {}


def _save_cache() -> None:
    """Persist embedding cache to disk."""
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(_embedding_cache, f)
    logger.info("Saved %d embeddings to cache %s", len(_embedding_cache), CACHE_PATH)


def get_embedding_dim() -> int:
    """Return the dimensionality of the code embeddings."""
    return EMBEDDING_DIM


def get_code_embedding(code_text: str) -> list:
    """Get a 768-dim dense embedding for a code string using CodeBERT.

    Uses mean-pooling over the last hidden state (ignoring padding tokens).
    Results are cached in memory and persisted to embedding_cache.json
    for fast reuse across runs.

    Args:
        code_text: The code string to embed (e.g., "os.system", "base64.b64decode").

    Returns:
        A list of 768 floats representing the dense embedding.
    """
    global _tokenizer, _model, _embedding_cache

    # Normalize the key
    cache_key: str = code_text.strip()

    # Check cache first
    _load_cache()
    if cache_key in _embedding_cache:
        return _embedding_cache[cache_key]

    # Lazy-load model
    _load_model()

    # Tokenize
    inputs = _tokenizer(
        cache_key,
        return_tensors="pt",
        max_length=MAX_SEQ_LENGTH,
        truncation=True,
        padding=True,
    ).to(_device)

    # Forward pass
    with torch.no_grad():
        outputs = _model(**inputs)
        # Mean pooling over non-padding tokens
        attention_mask = inputs["attention_mask"]
        mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
        sum_embeddings = torch.sum(outputs.last_hidden_state * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask

    # Convert to list
    embedding: list = mean_pooled.squeeze(0).cpu().numpy().tolist()

    # Cache it
    _embedding_cache[cache_key] = embedding
    _save_cache()

    return embedding


def extract_node_text(node: dict) -> str:
    """Extract meaningful text from a CPG node.

    Prefers the CODE field, falls back to NAME, then LABEL.

    Args:
        node: CPG node dict with fields like 'CODE', 'NAME', 'LABEL'.

    Returns:
        Best available text representation of the node.
    """
    code: str = node.get("CODE", "").strip()
    name: str = node.get("NAME", "").strip()
    label: str = node.get("LABEL", "").strip()

    # Prefer code text, but it can be empty for structural nodes
    if code and code not in ("<empty>", "<module>", "<global>"):
        return code
    if name and name not in ("<empty>", "<module>", "<global>"):
        return name
    # For的结构ural nodes, use label + name
    if name:
        return f"{label}:{name}"
    return label


def embed_graph_nodes(graph: dict) -> np.ndarray:
    """Embed all nodes in a graph, returning a [num_nodes, 768] array.

    Args:
        graph: Dict with 'nodes' list, each node having CODE/NAME/LABEL fields.

    Returns:
        numpy array of shape [num_nodes, EMBEDDING_DIM].
    """
    _load_model()
    _load_cache()

    embeddings: list = []
    for node in graph["nodes"]:
        text: str = extract_node_text(node)
        if not text:
            text = node.get("LABEL", "UNKNOWN")
        emb: list = get_code_embedding(text)
        embeddings.append(emb)

    return np.array(embeddings, dtype=np.float32)


# ---------------------------------------------------------------------------
# Module interface
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Quick test
    test_texts = ["os.system", "base64.b64decode", "def hello():", "import subprocess"]
    for text in test_texts:
        emb = get_code_embedding(text)
        print(f"  {text:30s} -> [{len(emb)}-dim] norm={np.linalg.norm(emb):.2f}")
    print(f"\nCache size: {len(_embedding_cache)} entries")
