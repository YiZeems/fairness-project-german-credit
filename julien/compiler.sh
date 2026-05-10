#!/bin/bash
# Compile all fairness-beamer documents (note d'analyse + 3 presentations).
# The 4 documents are aligned with responsiveAI-german-credit.ipynb.
#
# Usage:
#   bash compiler.sh        compile all 4 documents (clean before, then 2 passes each)
#   bash compiler.sh clean  remove auxiliary files only

set -e
cd "$(dirname "$0")"

DOCS=(
    note_analyse                            # 2-page analysis note (assignment requirement)
    fairness                                # short presentation, theoretical view
    explication-parcours-simple             # long detailed walkthrough
    explication-parcours-simple_beamer      # main slide deck with TikZ + outputs
)

clean() {
    echo "=== Cleaning auxiliary files ==="
    rm -f *.aux *.log *.nav *.out *.snm *.toc *.vrb *.fls *.fdb_latexmk *.synctex.gz preview-*.png note_preview*.png
    echo "Clean done."
}

compile_one() {
    local name="$1"
    if [ ! -f "${name}.tex" ]; then
        echo "[SKIP] ${name}.tex not found"
        return 1
    fi
    echo "--- Compiling ${name}.tex (pass 1/2) ---"
    "$LATEX" -interaction=nonstopmode "${name}.tex" > /dev/null
    echo "--- Compiling ${name}.tex (pass 2/2) ---"
    "$LATEX" -interaction=nonstopmode "${name}.tex" > /dev/null
    echo "[OK] ${name}.pdf"
}

if [ "${1:-}" = "clean" ]; then
    clean
    exit 0
fi

# Detect LaTeX compiler
if command -v pdflatex &>/dev/null; then
    LATEX=pdflatex
elif command -v xelatex &>/dev/null; then
    LATEX=xelatex
else
    echo "ERROR: No LaTeX compiler found (pdflatex or xelatex required)."
    exit 1
fi

echo "Using: $LATEX"
clean

# Compile each document
for doc in "${DOCS[@]}"; do
    compile_one "$doc"
done

# Final cleanup of aux files but keep PDFs
clean

echo ""
echo "=== Summary ==="
for doc in "${DOCS[@]}"; do
    if [ -f "${doc}.pdf" ]; then
        size=$(du -h "${doc}.pdf" | cut -f1)
        echo "  [OK] ${doc}.pdf (${size})"
    else
        echo "  [FAIL] ${doc}.pdf NOT generated"
    fi
done
