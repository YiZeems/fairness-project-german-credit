#!/bin/bash
# Compiler la présentation Beamer et la note d'analyse
cd "$(dirname "$0")"

clean() {
    rm -f *.aux *.log *.nav *.out *.snm *.toc *.vrb *.fls *.fdb_latexmk *.synctex.gz
    echo "Clean done."
}

compile_one() {
    local name="$1"
    if [ ! -f "${name}.tex" ]; then
        echo "SKIP: ${name}.tex not found"
        return 1
    fi
    echo "--- Compiling ${name}.tex ---"
    $LATEX -interaction=nonstopmode "${name}.tex"
    $LATEX -interaction=nonstopmode "${name}.tex"
    echo "Done: ${name}.pdf"
}

if [ "$1" = "clean" ]; then
    clean
    exit 0
fi

# Detect LaTeX compiler
if command -v pdflatex &>/dev/null; then
    LATEX=pdflatex
elif command -v xelatex &>/dev/null; then
    LATEX=xelatex
else
    echo "ERROR: No LaTeX compiler found."
    exit 1
fi

clean
echo "Using: $LATEX"

# Compile all files
compile_one explication-parcours-simple
compile_one explication-parcours-simple_beamer

echo "All done"
