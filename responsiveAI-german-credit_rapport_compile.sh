#!/bin/bash
# Compile responsiveAI-german-credit_rapport.tex (long pedagogical document)
# accompanying responsiveAI-german-credit_rendu.ipynb.
#
# Usage:
#   bash responsiveAI-german-credit_rapport_compile.sh        compile rapport (2 passes)
#   bash responsiveAI-german-credit_rapport_compile.sh clean  remove auxiliary files

set -e
cd "$(dirname "$0")"

DOC=responsiveAI-german-credit_rapport

clean() {
    echo "=== Cleaning auxiliary files ==="
    rm -f "${DOC}.aux" "${DOC}.log" "${DOC}.nav" "${DOC}.out" \
          "${DOC}.snm" "${DOC}.toc" "${DOC}.vrb" "${DOC}.fls" \
          "${DOC}.fdb_latexmk" "${DOC}.synctex.gz"
    echo "Clean done."
}

if [ "${1:-}" = "clean" ]; then
    clean
    exit 0
fi

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

if [ ! -f "${DOC}.tex" ]; then
    echo "ERROR: ${DOC}.tex not found"
    exit 1
fi

echo "--- Compiling ${DOC}.tex (pass 1/2) ---"
"$LATEX" -interaction=nonstopmode "${DOC}.tex" > /dev/null
echo "--- Compiling ${DOC}.tex (pass 2/2) ---"
"$LATEX" -interaction=nonstopmode "${DOC}.tex" > /dev/null
echo "[OK] ${DOC}.pdf"

clean

if [ -f "${DOC}.pdf" ]; then
    size=$(du -h "${DOC}.pdf" | cut -f1)
    echo ""
    echo "=== Result : ${DOC}.pdf (${size}) ==="
fi
