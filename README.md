<img width="130" height="130" alt="LLM_booster" src="https://github.com/user-attachments/assets/66e10a67-78a5-4854-87d9-e1acc88e8636" />


# LLM Test-Time Scaling Service

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A research framework for implementing and evaluating test-time scaling strategies for large language models. Includes implementations of DeepConf, Best-of-N, Self-Consistency, and Chain-of-Thought strategies.

---

## 🚀 Quick Start

### For Users (Run Experiments)

```bash
# 1. Create and activate conda environment
conda create -n lm-polygraph-env python=3.11 -y
conda activate lm-polygraph-env

# 2. Install base dependencies and lm-polygraph
./setup.sh

# 3. Install latex2sympy2 (must be installed separately due to antlr4 conflict)
pip install latex2sympy2 --no-deps

# 4. Install vLLM for fast local inference
pip install ".[vllm]"

# 5. Configure API keys
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY

# 6. Run DeepConf on GSM8K
python scripts/run_tts_eval.py \
  --config-name experiments/deepconf/run_gsm8k_deepconf_offline \
  dataset.subset=10
```

### Math Evaluation (MATH/Minerva datasets)

Math evaluation uses `latex2sympy2` for symbolic comparison (matching Qwen2.5-Math evaluation).
ANTLR version mismatch warnings may appear but functionality is correct:

```
ANTLR runtime and generated code versions disagree: 4.9.3!=4.7.2
```

This is expected - Hydra uses ANTLR 4.9.3, latex2sympy2 was built with 4.7.2. Both work correctly.

### For Developers (Contribute Code)

See [Onboarding Guide](#-onboarding-for-developers) below.

---

## 📚 Onboarding for Developers

**Welcome!** Follow these steps to get started with development:

### Step 1: Understand the Project

**Read the documentation:**
- **[Project Structure](docs/PROJECT_STRUCTURE.md)** - Architecture overview, components, design patterns
- **[Strategy Registration](docs/STRATEGY_REGISTRATION.md)** - How to add new strategies with tests
- **[DeepConf Guide](docs/deepconf/DeepConf.md)** - Example strategy implementation

**Quick architecture overview:**
```
llm_tts/strategies/     → TTS strategy implementations
llm_tts/models/         → Model wrappers with streaming support
llm_tts/scorers/        → Step scoring functions (PRM, uncertainty)
llm_tts/evaluation/     → Correctness evaluation methods
config/                 → Hydra configuration system
tests/                  → Test suite with strategy registry
```

### Step 2: Set Up Development Environment

```bash
# Clone the repository
git clone https://github.com/IINemo/llm-tts-service.git
cd llm-tts-service

# Create and activate conda environment
conda create -n lm-polygraph-env python=3.11 -y
conda activate lm-polygraph-env

# Install dependencies and lm-polygraph
./setup.sh

# Install latex2sympy2 (must be installed separately due to antlr4 conflict)
pip install latex2sympy2 --no-deps

# Install vLLM for fast local inference
pip install ".[vllm]"

# Install dev dependencies and git hooks
pip install -e ".[dev]"
make hooks
```

**What this does:**
- Creates isolated conda environment with Python 3.11
- Installs package in editable mode (`-e`)
- Installs lm-polygraph dev branch (for uncertainty estimation)
- Installs latex2sympy2 for math evaluation (--no-deps avoids antlr4 conflict with Hydra)
- Installs vLLM for fast local model inference
- Sets up pre-commit hooks (black, isort, flake8)

### Step 3: Configure API Keys

```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your keys:
# OPENROUTER_API_KEY=sk-or-v1-...
# DEEPSEEK_API_KEY=sk-...
```

**Required for:**
- Running experiments (OPENROUTER_API_KEY)
- Evaluation with LLM judge (DEEPSEEK_API_KEY or OPENROUTER_API_KEY)

### Step 4: Verify Installation

```bash
# Run tests to verify setup
pytest tests/strategy_registry.py --validate  # Validate registry
pytest tests/deepconf/ -v                     # Run DeepConf tests
pytest tests/online_best_of_n/ -v             # Run Best-of-N tests

# Or run all tests
make test
```

**Expected result:** All tests pass (some may skip if API keys not set)

### Step 5: Run Your First Experiment

```bash
# Quick test with 1 sample
python scripts/run_tts_eval.py \
  --config-name experiments/deepconf/run_gsm8k_deepconf_offline \
  dataset.subset=1 \
  strategy.budget=4

# Check the output
ls outputs/  # Results saved here with timestamp
```

### Step 6: Make Your First Change

**Example: Add a new strategy**

```bash
# 1. Create strategy file
touch llm_tts/strategies/strategy_my_new.py

# 2. Implement your strategy (inherit from StrategyBase)

# 3. Create tests
mkdir tests/my_new
touch tests/my_new/test_my_new.py

# 4. Register in strategy registry
# Edit tests/strategy_registry.py

# 5. Validate
python tests/strategy_registry.py --validate

# 6. Run tests
pytest tests/my_new/ -v
```

**See [Strategy Registration Guide](docs/STRATEGY_REGISTRATION.md) for detailed steps.**

### Step 7: Daily Development Workflow

```bash
# Make your changes...

# Format and check before committing
make fix     # Auto-fix with black, isort
make lint    # Check with flake8

# Run relevant tests
pytest tests/your_module/ -v

# Commit (hooks run automatically)
git commit -m "feat: add new feature"

# Push
git push origin your-branch
```

**Pre-commit hooks will:**
- Format code with black and isort
- Check for trailing whitespace, large files
- Run flake8 linting
- Block commit if checks fail

---

## 📁 Project Structure

```
llm-tts-service/
├── config/              # Hydra configuration (experiments, models, strategies)
├── llm_tts/             # Main library
│   ├── strategies/      # TTS strategy implementations (DeepConf, Best-of-N, etc.)
│   ├── models/          # Model wrappers (streaming, early stopping)
│   ├── scorers/         # Step scoring (PRM, uncertainty, voting)
│   ├── evaluation/      # Correctness evaluation (LLM judge, exact match)
│   └── datasets/        # Dataset utilities (GSM8K, etc.)
├── scripts/             # Main evaluation script (run_tts_eval.py)
├── tests/               # Test suite with strategy registry
├── docs/                # Documentation
└── lm-polygraph/        # Submodule: uncertainty estimation
```

**Quick Overview:**
- **Strategies**: DeepConf (confidence-based), Best-of-N (PRM scoring), Self-Consistency, Chain-of-Thought
- **Configuration**: Hierarchical Hydra configs - see `config/README.md`
- **Evaluation**: Two-phase pipeline (generation → evaluation) with multi-evaluator support
- **Testing**: Strategy registry enforces test coverage - see [Strategy Registration Guide](docs/STRATEGY_REGISTRATION.md)

**📖 For detailed architecture and component descriptions, see [Project Structure Documentation](docs/PROJECT_STRUCTURE.md)**

---

## 🔧 Development Commands

```bash
# Testing
make test              # Run all tests
pytest tests/path/ -v  # Run specific tests

# Code Quality
make fix               # Auto-fix formatting (black, isort)
make format            # Format only (no other hooks)
make lint              # Check with flake8
make hooks             # Install pre-commit hooks

# Validation
python tests/strategy_registry.py --validate  # Validate all strategies
python tests/strategy_registry.py --list      # List registered strategies
```

## 💡 Usage Examples

**Note:** API keys are loaded from `.env` file - no need to specify them in the command.

### DeepConf Strategy

**Offline mode** (generate N traces, filter by confidence, majority vote):
```bash
python scripts/run_tts_eval.py \
  --config-name experiments/deepconf/run_gsm8k_deepconf_offline \
  model.model_path="openai/gpt-3.5-turbo" \
  dataset.subset=10
```

**Online mode** (adaptive generation with confidence-based early stopping):
```bash
python scripts/run_tts_eval.py \
  --config-name experiments/deepconf/run_gsm8k_deepconf_online \
  model.model_path="openai/gpt-3.5-turbo" \
  dataset.subset=10
```

### Best-of-N Strategy

With Qwen-3 (local model):
```bash
WANDB_ENTITY=nlpresearch.group WANDB_PROJECT=tts python scripts/run_tts_eval.py \
  --config-path ../config \
  --config-name run_tts_eval \
  dataset=small_gsm8k \
  dataset.subset=1 \
  model=hf_qwen3
```

With ChatGPT via OpenRouter:
```bash
WANDB_ENTITY=nlpresearch.group WANDB_PROJECT=tts python scripts/run_tts_eval.py \
  --config-path ../config \
  --config-name run_tts_eval \
  dataset=small_gsm8k \
  dataset.subset=1 \
  model=openai \
  model.model_path="gpt-4o-mini"
```

With uncertainty scorer:
```bash
WANDB_ENTITY=nlpresearch.group WANDB_PROJECT=tts python scripts/run_tts_eval.py \
  --config-path ../config \
  --config-name run_tts_eval \
  dataset=small_gsm8k \
  dataset.subset=1 \
  model=hf_qwen3 \
  scorer=uncertainty
```

---

## 🛡️ Robustness & Resume Features

**Evaluations are crash-resistant** - results saved after each sample, resume from interruptions with `--resume`.

### Key Features

- ✅ **Incremental Saving**: No work lost - saves after each sample (not batched)
- ✅ **Resume Capability**: Continue from where you left off with `--resume` or `--resume-from`
- ✅ **Full Reproducibility**: Every run saves complete config snapshot in `.hydra/`

### Quick Resume

```bash
# Resume from latest
python scripts/run_tts_eval.py --config-name your_experiment --resume

# Resume from specific run
python scripts/run_tts_eval.py --resume-from outputs/2025-10-18/23-50-46
```

### Evaluation
To re-evaluate or use another evaluator for already existing run use `scripts/evaluate_results.py` providing path to run's folder.

To re-evaluate using same config:
```
python3 scripts/evaluate_results.py --results_dir outputs_path
```
To add and override config.evaluation.evaluators pass `--evaluators` flag and list them, e.g.:
```
python3 scripts/evaluate_results.py --results_dir outputs_path --evaluators exact_match llm_judge
```
You can modify configs for new evaluators by passing `--llm_judge_config` or `--alignscore_config`, default is used if no config is passed, but evaluator is called.

By default, script saves new file back to the same folder instead of original file. If it is not desired you can pass `--save-to` argument to specify another folder.

**📖 For detailed documentation, troubleshooting, and best practices, see [Robustness Guide](docs/ROBUSTNESS.md)**

---

## 🌐 REST API Service (Optional)

Deploy strategies as a REST API for production use.

### Quick Start

```bash
./start_service_app.sh  # Automated setup with Docker
```

### Manual Setup

```bash
# With Docker
export OPENROUTER_API_KEY="your-key"
docker-compose up -d

# Without Docker (local dev)
pip install -e ".[service]"
export OPENROUTER_API_KEY="your-key"
python service_app/main.py
```

**See `service_app/README.md` for API documentation.**

---

## 🔧 Troubleshooting

### vLLM engine fails to start

If you see `RuntimeError: Engine core initialization failed` or `Bytes object is corrupted, checksum does not match`, the vLLM torch compile cache is corrupted. Clear it:

```bash
rm -rf ~/.cache/vllm/torch_compile_cache/
```

---

## 📖 Documentation

- **[Project Structure](docs/PROJECT_STRUCTURE.md)** - Detailed architecture and components
- **[Evaluation Protocol](docs/evaluation/README.md)** - Datasets, metrics (accuracy, tokens, FLOPs), and reporting guidelines
- **[Strategy Registration](docs/STRATEGY_REGISTRATION.md)** - Adding new strategies with tests
- **[Robustness Guide](docs/ROBUSTNESS.md)** - Incremental saving, resume, and reproducibility
- **[DeepConf Guide](docs/deepconf/DeepConf.md)** - Confidence-based test-time scaling
- **[GSM8K Dataset](docs/datasets/GSM8K/)** - Dataset usage examples
- **[Configuration Guide](config/README.md)** - Hydra config system

---

## 🤝 Contributing

1. Read the [Onboarding Guide](#-onboarding-for-developers)
2. Check [Strategy Registration](docs/STRATEGY_REGISTRATION.md) for requirements
3. Follow the [Daily Workflow](#step-7-daily-development-workflow)
4. Ensure tests pass: `make test`
5. Submit a PR

---

## 📝 TODO

- Add new scorers (semantic similarity, calibration-based)
- Implement Tree of Thought strategy
- Add MATH dataset support

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
