#!/bin/bash
# batch_extract.sh — Batch CPG extraction using bash (avoids Python subprocess pipe issues)
# Usage: ./batch_extract.sh <dataset_dir> <output_dir>

set -euo pipefail

DATASET_DIR="${1:-dataset}"
OUTPUT_DIR="${2:-raw_graphs}"
JOERN_DIR="$HOME/bin/joern"
JAVA_HOME="$HOME/.local/jdk/jdk-21.0.5+11"

export JAVA_HOME
export PATH="$JAVA_HOME/bin:$PATH"

JOERN_PARSE="$JOERN_DIR/bin/joern-parse"
JOERN_EXPORT="$JOERN_DIR/bin/joern-export"

mkdir -p "$OUTPUT_DIR"

total=0
success=0
fail=0

for py_file in "$DATASET_DIR"/benign/*.py "$DATASET_DIR"/malicious/*.py; do
    [ -f "$py_file" ] || continue
    total=$((total + 1))
    
    basename=$(basename "$py_file" .py)
    label=$(basename "$(dirname "$py_file")")
    output_json="$OUTPUT_DIR/${label}_${basename}.json"
    tmp_cpg="/tmp/cpg_${basename}_$$.bin"
    export_dir="/tmp/dot_${basename}_$$"
    
    echo "[$total] $label/$basename.py"
    
    # Clean stale
    rm -f "$tmp_cpg"
    rm -rf "$export_dir"
    
    # Parse
    if ! "$JOERN_PARSE" "$py_file" --output "$tmp_cpg" >/dev/null 2>&1; then
        echo "  -> PARSE FAILED"
        fail=$((fail + 1))
        rm -f "$tmp_cpg"
        continue
    fi
    
    if [ ! -f "$tmp_cpg" ]; then
        echo "  -> CPG NOT CREATED"
        fail=$((fail + 1))
        continue
    fi
    
    # Export
    if ! "$JOERN_EXPORT" "$tmp_cpg" --out "$export_dir" --repr cpg --format dot >/dev/null 2>&1; then
        echo "  -> EXPORT FAILED"
        fail=$((fail + 1))
        rm -f "$tmp_cpg"
        rm -rf "$export_dir"
        continue
    fi
    
    # Consolidate DOT -> JSON
    python3 << PYEOF > "$output_json"
import json, re, os
from pathlib import Path

export_dir = Path("$export_dir")
nodes = {}
edges = []

node_re = re.compile(r'"(\d+)"\s*\[([^\]]+)\]')
edge_re = re.compile(r'"(\d+)"\s*->\s*"(\d+)"\s*\[([^\]]+)\]')
attr_re = re.compile(r'(\w+)="([^"]*)"')

for dot_file in export_dir.rglob("export.dot"):
    content = dot_file.read_text()
    for m in node_re.finditer(content):
        nid = int(m.group(1))
        attrs = dict(attr_re.findall(m.group(2)))
        if nid not in nodes:
            nodes[nid] = {"id": nid, **attrs}
    for m in edge_re.finditer(content):
        attrs = dict(attr_re.findall(m.group(3)))
        edges.append({"src": int(m.group(1)), "dst": int(m.group(2)), **attrs})

json.dump({"nodes": list(nodes.values()), "edges": edges}, open("$output_json", "w"), indent=2)
print(f"{len(nodes)} nodes, {len(edges)} edges")
PYEOF
    
    echo "  -> SUCCESS"
    success=$((success + 1))
    
    # Cleanup
    rm -f "$tmp_cpg"
    rm -rf "$export_dir"
done

echo ""
echo "============================================"
echo "Batch extraction complete: $success succeeded, $fail failed out of $total total"
echo "Raw graphs saved to: $OUTPUT_DIR"
