#!/usr/bin/env python3
"""
web_app.py — Gradio Web UI for Semantic CPG Backdoor Detector.

Provides a browser-based interface for analyzing Python code snippets
for supply chain backdoor patterns. No command-line knowledge required.

Usage:
    python web_app.py
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import gradio as gr

from explain import analyze_code


def format_result(result: dict) -> str:
    """Convert analyze_code() dictionary to formatted Markdown.

    Args:
        result: Dictionary from analyze_code() with keys:
            verdict, confidence, explanation_nodes, explanation_flows

    Returns:
        Markdown-formatted string for Gradio display.
    """
    verdict: str = result.get("verdict", "ERROR")
    confidence: float = result.get("confidence", 0.0)
    nodes: list[dict] = result.get("explanation_nodes", [])
    flows: list[str] = result.get("explanation_flows", [])

    lines: list[str] = []

    if verdict == "MALICIOUS":
        lines.append("## 🚨 MALICIOUS CODE DETECTED")
        lines.append("")
        lines.append(f"**Confidence:** {confidence:.1f}%")
        lines.append("")
        lines.append("The Graph Neural Network detected semantic patterns consistent with")
        lines.append("obfuscated backdoor code by analyzing the Code Property Graph of the")
        lines.append("supplied Python source.")
    elif verdict == "BENIGN":
        lines.append("## ✅ CODE APPEARS SAFE")
        lines.append("")
        lines.append(f"**Confidence:** {confidence:.1f}%")
        lines.append("")
        lines.append("No backdoor semantic patterns were detected in the supplied code.")
    else:
        lines.append("## ⚠️ ANALYSIS ERROR")
        lines.append("")
        lines.append("An error occurred during analysis. Please check the input and try again.")
        return "\n".join(lines)

    if nodes:
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("### 🔍 Critical Code Elements")
        lines.append("")
        for i, node in enumerate(nodes, 1):
            score = node.get("score", 0.0)
            code = node.get("code", "?")
            ntype = node.get("type", "UNKNOWN")
            lines.append(f"{i}. ⚠️ **`{code}`** (Score: {score:.2f}, Type: {ntype})")

    if flows:
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("### 🔗 Critical Data Flows")
        lines.append("")
        for flow in flows:
            lines.append(f"➡️ {flow}")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("_Powered by CodeBERT + Graph Convolutional Networks + GNNExplainer_")

    return "\n".join(lines)


def analyze_pasted_code(code_text: str) -> str:
    """Analyze pasted Python code and return formatted Markdown.

    Saves the code to a temporary file, runs analyze_code(), and cleans up.

    Args:
        code_text: Raw Python source code pasted by the user.

    Returns:
        Formatted Markdown string with verdict and explanation.
    """
    if not code_text or not code_text.strip():
        return "## ⚠️ No code provided\n\nPlease paste Python code in the text area above."

    # Write to temp file
    tmp_file = Path(tempfile.gettempdir()) / f"gradio_upload_{os.getpid()}.py"
    try:
        tmp_file.write_text(code_text, encoding="utf-8")
        result = analyze_code(str(tmp_file))
        return format_result(result)
    except Exception as e:
        return f"## ⚠️ Analysis Error\n\n```\n{type(e).__name__}: {e}\n```"
    finally:
        if tmp_file.exists():
            tmp_file.unlink()


# ---------------------------------------------------------------------------
# Gradio Interface
# ---------------------------------------------------------------------------

with gr.Blocks(
    title="Semantic CPG Backdoor Detector",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown(
        "# 🔬 Semantic CPG Backdoor Detector\n"
        "Paste Python code below to analyze it for supply chain backdoor patterns "
        "using **CodeBERT + Graph Neural Networks + GNNExplainer**.\n\n"
        "_Inspired by real-world attacks like XZ Utils (CVE-2024-3094)._"
    )

    with gr.Row():
        with gr.Column(scale=1):
            code_input = gr.Textbox(
                label="Paste Python Code to Analyze",
                lines=15,
                placeholder="def suspicious_function():\n    import os\n    ...",
                max_lines=30,
            )
            analyze_btn = gr.Button("🔍 Analyze Code", variant="primary")

        with gr.Column(scale=1):
            result_output = gr.Markdown(
                value="## ⏳ Waiting for input\n\nPaste code and click **Analyze Code** to begin.",
            )

    analyze_btn.click(
        fn=analyze_pasted_code,
        inputs=code_input,
        outputs=result_output,
    )

    gr.Markdown(
        "---\n"
        "**How it works:** Source Code → Code Property Graph (Joern) → "
        "CodeBERT Embeddings (768-dim) → 3-Layer GCN → Verdict + Explanation"
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
