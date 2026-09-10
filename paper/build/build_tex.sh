#!/usr/bin/env bash
# Rebuild the IEEE two-column LaTeX manuscript for the TrustNet-RCA paper.
# Usage: ./build_tex.sh [output.tex]
#
# Shares src/ and splice.py with build.sh, so the .tex and the .docx are
# rendered from one manuscript and cannot drift apart.
set -euo pipefail
cd "$(dirname "$0")"

OUT="${1:-../Trustworthy_RCA_Framework_Paper_IEEE.tex}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

command -v pandoc >/dev/null || { echo "error: pandoc not found" >&2; exit 1; }

echo "1/2 collecting sections"
python3 splice.py src "$WORK/paper.md"

echo "2/2 rendering IEEEtran LaTeX"
python3 to_latex.py "$WORK/paper.md" src/frontmatter.md "$OUT"

echo "done -> $OUT"
echo
if command -v latexmk >/dev/null; then
    echo "compile with:  (cd $(dirname "$OUT") && latexmk -pdf $(basename "$OUT"))"
elif command -v pdflatex >/dev/null; then
    echo "compile with:  (cd $(dirname "$OUT") && pdflatex $(basename "$OUT") && pdflatex $(basename "$OUT"))"
else
    echo "note: no LaTeX toolchain here. Install one (Debian/Ubuntu):"
    echo "        sudo apt install texlive-latex-recommended texlive-publishers \\"
    echo "                         texlive-science texlive-fonts-recommended latexmk"
    echo "      or upload the .tex plus figures/ to Overleaf."
fi
