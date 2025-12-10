# Datasets

All datasets are available in our HuggingFace collection: [test-time-compute](https://huggingface.co/test-time-compute)

## Mathematical Reasoning

| Dataset | Size | Answer Type | Release | Notes |
|---------|------|-------------|---------|-------|
| **AIME 2024** | 30 | Integer (0-999) | 2024 | Competition problems, very hard |
| **AIME 2025** | 30 | Integer (0-999) | 2025 | Competition problems, very hard |
| **MATH-500** | 500 | Numeric/Expression | 2023 | Subset from "Let's Verify Step by Step" (OpenAI) |
| **SVAMP** | 1000 | Integer | 2021 | Real-world arithmetic problems |

---

## Dataset Details

### AIME 2024 / AIME 2025

American Invitational Mathematics Examination problems.

- **Source**: [test-time-compute/aime_2024](https://huggingface.co/datasets/test-time-compute/aime_2024), [test-time-compute/aime_2025](https://huggingface.co/datasets/test-time-compute/aime_2025)
- **Answer format**: Integer between 0 and 999
- **Difficulty**: Very hard (competition-level)
- **Widely used** in test-time compute papers

```yaml
dataset:
  name: test-time-compute/aime_2025
  split: test
  question_field: problem
  answer_field: answer
```

### MATH-500

Subset of MATH dataset created by OpenAI for the "Let's Verify Step by Step" paper.

- **Source**: Original MATH dataset with OpenAI's subset selection
- **Answer format**: Numeric or mathematical expression
- **Requires**: Math parsing for exact match comparison
- **Note**: Full MATH dataset (released 2021) may be saturated

### SVAMP

Simple Variations on Arithmetic Math word Problems.

- **Source**: [SVAMP dataset](https://github.com/arkilpatel/SVAMP)
- **Answer format**: Always an integer
- **Difficulty**: Elementary to middle school level
- **Use case**: Baseline evaluation, sanity checks

---

## Dataset Loading Example

```python
from datasets import load_dataset

# AIME 2025
dataset = load_dataset("test-time-compute/aime_2025", split="test")

# Access fields
for sample in dataset:
    question = sample["problem"]
    answer = sample["answer"]  # Integer string
```

---

## Notes

- Some strategies were originally developed for code reasoning tasks and evaluated on code validity, not math.
- Some papers used custom/crafted datasets not publicly available.
- We focus on publicly available math reasoning benchmarks for reproducibility.
