# Midterm Report

This directory contains the midterm report for the NLP 701 project on Test-Time Scaling for LLM Reasoning.

## Files

- `midterm_report.tex` - Main LaTeX source file
- `references.bib` - Bibliography with all cited papers
- `acl.sty` - ACL conference style file
- `acl_natbib.bst` - ACL bibliography style
- `compile.sh` - Compilation script

## Compilation

To compile the report, run:

```bash
cd NLP_project/midterm
./compile.sh
```

Or manually:

```bash
pdflatex midterm_report.tex
bibtex midterm_report
pdflatex midterm_report.tex
pdflatex midterm_report.tex
```

The output will be `midterm_report.pdf`.

## Content Overview

The report includes:

1. **Introduction** - Overview of test-time scaling and contributions
2. **Background** - Chain-of-Thought prompting and test-time scaling concepts
3. **Related Work** - Self-Consistency, confidence-based selection, iterative refinement, and uncertainty quantification
4. **Methods** - Detailed description of Self-Consistency and DeepConf implementations
5. **Experiments** - Experimental setup, datasets (GSM8K and AIME 2025), and results
6. **Conclusion and Future Work** - Summary of findings and future directions

## Key Results

- **GSM8K**: Both Self-Consistency and DeepConf achieve 93.3% accuracy
- **AIME 2025**: DeepConf (16.7%) outperforms Self-Consistency (7.7%) by 9 percentage points
- **Service**: OpenAI-compatible REST API for practical deployment

## Experimental Data

The experimental results are based on:
- 30 problems from GSM8K test set
- 30 problems from AIME 2025
- GPT-4o-mini for generation (via OpenRouter)
- DeepSeek-R1-0528 for evaluation
- Budget of 16 reasoning traces per problem
