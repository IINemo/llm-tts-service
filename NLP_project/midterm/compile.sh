#!/bin/bash
# Compile the midterm report

echo "Compiling midterm report..."

# Run pdflatex
pdflatex midterm_report.tex

# Run bibtex
bibtex midterm_report

# Run pdflatex twice more for references
pdflatex midterm_report.tex
pdflatex midterm_report.tex

echo "Done! Output: midterm_report.pdf"

# Clean up auxiliary files
rm -f *.aux *.bbl *.blg *.log *.out
