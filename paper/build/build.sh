#!/usr/bin/env bash
# Rebuild the IEEE two-column DOCX for the TrustNet-RCA paper.
# Usage: ./build.sh [output.docx]
set -euo pipefail
cd "$(dirname "$0")"

OUT="${1:-../Trustworthy_RCA_Framework_Paper_IEEE.docx}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

command -v pandoc >/dev/null || { echo "error: pandoc not found" >&2; exit 1; }
python3 -c 'import docx' 2>/dev/null || { echo "error: python-docx not installed (pip install python-docx)" >&2; exit 1; }

echo "1/5 building IEEE style template"
python3 make_ref.py "$WORK/ieee-reference.docx"

echo "2/5 splicing manuscript sections"
python3 splice.py src "$WORK/paper_v2.md"

echo "3/5 applying IEEE section numbering"
python3 transform.py "$WORK/paper_v2.md" "$WORK/body.md" "$WORK/front.json"
cp src/title_override.txt "$WORK/"

echo "4/5 converting to docx"
# --resource-path=.. resolves the figures/ references against paper/.
pandoc "$WORK/body.md" --reference-doc="$WORK/ieee-reference.docx" \
       --resource-path=".." -o "$WORK/body.docx"

echo "5/5 assembling title block, full-width spans, table layout"
python3 assemble.py "$WORK/body.docx" "$WORK/front.json" "$OUT"

echo "done -> $OUT"
