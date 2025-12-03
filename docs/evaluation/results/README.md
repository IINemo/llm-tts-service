# Experiment Results

This folder contains experimental results organized by dataset.

## Structure

```
results/
├── README.md           # This file
├── aime2025.md         # AIME 2025 results
├── gsm8k.md            # GSM8K results (future)
└── math500.md          # MATH-500 results (future)
```

## Adding New Results

When adding results for a new experiment:

1. Use the appropriate dataset file (create if needed)
2. Include the following information:
   - Strategy name and key hyperparameters
   - Model used
   - Accuracy, token count, FLOPs (see [metrics.md](../metrics.md))
   - WandB run link (if available)
   - Output directory path
   - Any observations or notes

## Result Entry Template

```markdown
### [Strategy Name] - [Model] - [Date]

| Metric | Value |
|--------|-------|
| Accuracy | XX.X% (N/M correct) |
| Avg Tokens/Sample | X,XXX |
| Avg TFLOPs/Sample | X.XX |
| Total Samples | N |

**Configuration**: `config/experiments/[strategy]/[config].yaml`

**WandB**: [run-name](https://wandb.ai/...)

**Output**: `outputs/YYYY-MM-DD/[run-dir]/`

**Notes**:
- Key observations
- Issues encountered
```

## Aggregating Results

To compare strategies, create summary tables in each dataset file showing side-by-side comparisons.
