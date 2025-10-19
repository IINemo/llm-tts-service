# Datasets for Test-Time Scaling

This directory documents the datasets supported by the TTS evaluation framework and how to add new ones.

## Supported Datasets

| Dataset | HuggingFace Link | Problems | Difficulty | Status | Description |
|---------|------------------|----------|------------|--------|-------------|
| **GSM8K** | [test-time-compute/test_gsm8k](https://huggingface.co/datasets/test-time-compute/test_gsm8k) | 1,319 | Grade School | ✅ Configured | Grade school math word problems |
| **AIME 2025** | [test-time-compute/aime_2025](https://huggingface.co/datasets/test-time-compute/aime_2025) | 30 | Competition | ✅ Configured | American Invitational Mathematics Examination |
| **MATH** | [test-time-compute/test_MATH](https://huggingface.co/datasets/test-time-compute/test_MATH) | 5,000 | High School+ | 🔜 Available | Competition mathematics problems (MATH dataset) |
| **Proofnet** | [test-time-compute/test_Proofnet](https://huggingface.co/datasets/test-time-compute/test_Proofnet) | 186 | Proof-based | 🔜 Available | Formal proof generation problems |

---

## Unified Dataset Format

All datasets in the `test-time-compute` organization follow a **unified format** for consistency across TTS experiments:

```json
{
  "question": "Problem statement as string",
  "answer": "Final answer only (no reasoning)",
  "metadata": {
    "dataset": "dataset_name",
    "problem_idx": 123,
    "problem_type": "optional_category",
    "difficulty": "optional_level",
    "original_answer": "optional_original_field"
  }
}
```

### Field Descriptions

- **`question`** (required): The problem statement or query
- **`answer`** (required): The gold standard answer, cleaned to just the final result
- **`metadata`** (required): Dictionary containing:
  - `dataset`: Source dataset name (e.g., "gsm8k", "aime_2025")
  - `problem_idx`: Problem index or ID
  - `problem_type`: Optional category (e.g., "Number Theory", "Geometry")
  - `difficulty`: Optional level (e.g., "grade_school", "competition")
  - `original_answer`: Optional original answer field if it contained reasoning

### Benefits of Unified Format

1. **Consistent evaluation**: Same code works across all datasets
2. **Easy integration**: New datasets plug in without code changes
3. **Metadata tracking**: Preserve source information for analysis
4. **Answer extraction**: Standardized format simplifies parsing

---

## Using Datasets

### Quick Start

```python
from datasets import load_dataset

# Load from HuggingFace
dataset = load_dataset("test-time-compute/aime_2025", split="train")

# Access examples
for example in dataset:
    question = example["question"]
    answer = example["answer"]
    metadata = example["metadata"]
```

### In Evaluation Pipelines

```bash
# GSM8K evaluation
python scripts/run_tts_eval.py \
  --config-name experiments/deepconf/run_gsm8k_deepconf_offline \
  dataset.subset=10

# AIME 2025 evaluation
python scripts/run_tts_eval.py \
  --config-name experiments/deepconf/run_aime_2025_deepconf_offline \
  dataset.subset=5
```

---

## Adding a New Dataset

Follow these steps to add a new dataset to the TTS framework:

### 1. Convert to Unified Format

Use the conversion script:

```bash
# Add converter function to scripts/convert_datasets.py
def convert_your_dataset(split: str) -> List[Dict]:
    """Convert YOUR_DATASET to unified format."""
    dataset = load_dataset("source/your_dataset", split=split)

    unified_data = []
    for example in dataset:
        unified_example = {
            "question": example["problem_field"],  # Adjust field name
            "answer": extract_answer(example["answer_field"]),
            "metadata": {
                "dataset": "your_dataset",
                "problem_idx": example["id"],
                "difficulty": "your_difficulty",
                # Add other relevant metadata
            },
        }
        unified_data.append(unified_example)

    return unified_data

# Register in DATASET_CONVERTERS
DATASET_CONVERTERS = {
    # ... existing converters
    "your_dataset": convert_your_dataset,
}
```

### 2. Convert the Dataset

```bash
python scripts/convert_datasets.py --dataset your_dataset
```

This creates:
- `data/unified/your_dataset_train.jsonl` (JSONL format)
- `data/unified/your_dataset_train.json` (JSON format)
- `data/unified/your_dataset_train_metadata.json` (Metadata)

### 3. Upload to HuggingFace

```bash
# Test upload first
python scripts/upload_to_hf.py \
  --dataset your_dataset \
  --org test-time-compute \
  --dry-run

# Actual upload (requires org permissions)
python scripts/upload_to_hf.py \
  --dataset your_dataset \
  --org test-time-compute
```

### 4. Create Configuration Files

Create three config files:

**a) Dataset config:** `config/dataset/your_dataset.yaml`
```yaml
dataset_path: "test-time-compute/your_dataset"
dataset_config: "default"
dataset_split: "train"
subset: null
prompt_file: "./config/prompts/your_dataset_deepconf.txt"
```

**b) Prompt template:** `config/prompts/your_dataset_deepconf.txt`
```
Problem: {question}

[Add dataset-specific instructions here]
Put your final answer in \boxed{{}}.
```

**c) Experiment config:** `config/experiments/deepconf/run_your_dataset_deepconf_offline.yaml`
```yaml
# @package _global_

defaults:
  - /dataset/your_dataset
  - /model/openrouter
  - /generation/default
  - /system/default
  - _self_

verbose: false
report_to: wandb
wandb_project: llm-tts-eval-deepconf

model:
  provider: openrouter
  model_name: "openai/gpt-4o-mini"
  top_logprobs: 20

generation:
  max_new_tokens: 2048  # Adjust based on problem complexity
  temperature: 0.7

scorer: null

strategy:
  type: deepconf
  mode: "offline"
  budget: 16  # Adjust based on difficulty
  window_size: 2048
  temperature: 0.6
  top_p: 0.95
  filter_method: "top5"

evaluation:
  evaluators:
    - llm_judge
  llm_judge:
    provider: openrouter
    base_url: https://openrouter.ai/api/v1
    model: deepseek/deepseek-r1-0528

dataset:
  subset: 5  # Adjust for testing
```

### 5. Test the Pipeline

```bash
# Smoke test with 1 sample
python scripts/run_tts_eval.py \
  --config-name experiments/deepconf/run_your_dataset_deepconf_offline \
  dataset.subset=1

# Check results
cat outputs/YYYY-MM-DD/HH-MM-SS/results.json
```

### 6. Document the Dataset

Add entry to this README's "Supported Datasets" table with:
- Dataset name
- HuggingFace link
- Number of problems
- Difficulty level
- Brief description

---

## Dataset Conversion Script

The `scripts/convert_datasets.py` utility handles conversion to unified format:

**Features:**
- Automatic format conversion
- Answer extraction/cleaning
- Metadata preservation
- Multiple output formats (JSON, JSONL)
- Schema validation

**Usage:**
```bash
# Convert dataset
python scripts/convert_datasets.py --dataset aime_2025

# Specify output directory
python scripts/convert_datasets.py \
  --dataset your_dataset \
  --output_dir custom/path

# Specify split
python scripts/convert_datasets.py \
  --dataset your_dataset \
  --split test
```

---

## HuggingFace Upload Script

The `scripts/upload_to_hf.py` utility uploads datasets to HuggingFace Hub:

**Features:**
- Dry-run mode for testing
- Automatic dataset card generation
- Metadata validation
- Error handling with troubleshooting

**Usage:**
```bash
# Dry run (preview without uploading)
python scripts/upload_to_hf.py \
  --dataset aime_2025 \
  --org test-time-compute \
  --dry-run

# Actual upload
python scripts/upload_to_hf.py \
  --dataset aime_2025 \
  --org test-time-compute
```

**Prerequisites:**
1. Login to HuggingFace: `huggingface-cli login`
2. Get write access to `test-time-compute` organization
3. Verify dataset exists in `data/unified/`

---

## Dataset Card Template

When uploading to HuggingFace, datasets get an auto-generated card with:

- License and task categories
- Dataset description and structure
- Example usage code
- Field explanations
- Citation information

See example: [test-time-compute/aime_2025](https://huggingface.co/datasets/test-time-compute/aime_2025)

---

## Troubleshooting

### Dataset Loading Issues

**Problem**: `DatasetNotFoundError`
```
Solution: Verify dataset path and split in config/dataset/your_dataset.yaml
```

**Problem**: Missing fields in loaded data
```
Solution: Check unified format conversion in convert_datasets.py
```

### Upload Issues

**Problem**: `403 Forbidden` when uploading
```
Solution:
1. Run: huggingface-cli login
2. Request write access to test-time-compute organization
3. Verify you have permissions at https://huggingface.co/test-time-compute
```

**Problem**: Dataset already exists
```
Solution: Either use a different name or delete existing dataset first
```

---

## Contributing

When adding datasets:

1. **Follow the unified format** - Ensures consistency
2. **Clean answers** - Extract only final answer, no reasoning
3. **Preserve metadata** - Keep source information
4. **Document thoroughly** - Update this README
5. **Test the pipeline** - Run smoke test before committing
6. **Use descriptive names** - Follow convention: `{source}_{year}` or `{source}_{variant}`

---

## Additional Resources

- [Project Structure](../PROJECT_STRUCTURE.md)
- [Strategy Registration](../STRATEGY_REGISTRATION.md)
- [DeepConf Guide](../deepconf/DeepConf.md)
- [HuggingFace Datasets Docs](https://huggingface.co/docs/datasets)
