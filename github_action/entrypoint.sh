#!/bin/bash
# entrypoint.sh — GitHub Action entrypoint for Semantic Backdoor Detector
set -euo pipefail

TARGET_PATH="${INPUT_TARGET_PATH:-.}"

echo "============================================"
echo "  Semantic CPG Backdoor Detector v2.2"
echo "  Scanning: ${TARGET_PATH}"
echo "============================================"
echo ""

# If target is a directory, find all .py files
if [ -d "$TARGET_PATH" ]; then
    echo "[*] Directory detected, scanning all .py files..."
    FILES=$(find "$TARGET_PATH" -name "*.py" -type f)
    if [ -z "$FILES" ]; then
        echo "[!] No Python files found in ${TARGET_PATH}"
        echo "result=BENIGN" >> "$GITHUB_OUTPUT"
        exit 0
    fi
    MALICIOUS_FOUND=false
    for f in $FILES; do
        echo "---"
        echo "[*] Analyzing: $f"
        RESULT=$(python explain.py --target "$f" --json 2>/dev/null || echo '{"verdict":"ERROR"}')
        echo "$RESULT"
        VERDICT=$(echo "$RESULT" | jq -r '.verdict // "ERROR"')
        if [ "$VERDICT" = "MALICIOUS" ]; then
            MALICIOUS_FOUND=true
        fi
    done
    if [ "$MALICIOUS_FOUND" = true ]; then
        echo "result=MALICIOUS" >> "$GITHUB_OUTPUT"
        exit 1
    else
        echo "result=BENIGN" >> "$GITHUB_OUTPUT"
        exit 0
    fi
fi

# Single file analysis
echo "[*] Running analysis..."
RESULT=$(python explain.py --target "$TARGET_PATH" --json 2>/dev/null || echo '{"verdict":"ERROR","confidence":0}')

# Print formatted output
echo ""
echo "$RESULT" | jq .

# Extract values for GitHub Actions output
VERDICT=$(echo "$RESULT" | jq -r '.verdict // "ERROR"')
CONFIDENCE=$(echo "$RESULT" | jq -r '.confidence // 0')
NODES=$(echo "$RESULT" | jq -c '.explanation_nodes // []')
FLOWS=$(echo "$RESULT" | jq -c '.explanation_flows // []')

# Set outputs
echo "verdict=${VERDICT}" >> "$GITHUB_OUTPUT"
echo "confidence=${CONFIDENCE}" >> "$GITHUB_OUTPUT"
echo "explanation-nodes=${NODES}" >> "$GITHUB_OUTPUT"
echo "explanation-flows=${FLOWS}" >> "$GITHUB_OUTPUT"

echo ""
echo "============================================"
echo "  VERDICT: ${VERDICT} (${CONFIDENCE}%)"
echo "============================================"

# Exit with appropriate code
if [ "$VERDICT" = "MALICIOUS" ]; then
    exit 1
fi
exit 0
