# Contributing to ThinkBooster

Thank you for your interest in contributing! This guide covers environment setup, development workflow, and how to add new components.

---

## Step 1: Understand the Project

Start by reading the core documentation:

- **[Project Structure](PROJECT_STRUCTURE.md)** — architecture overview, components, and design patterns
- **[Strategy Registration](STRATEGY_REGISTRATION.md)** — how to add new strategies with tests
- **[DeepConf Guide](deepconf/DeepConf.md)** — example strategy implementation

Quick architecture overview:

```
llm_tts/strategies/     → TTS strategy implementations
llm_tts/models/         → Model wrappers with streaming support
llm_tts/scorers/        → Step scoring functions (PRM, uncertainty)
llm_tts/evaluation/     → Correctness evaluation methods
config/                 → Hydra configuration system
tests/                  → Test suite with strategy registry
```

---

## Step 2: Set Up Development Environment

```bash
# Clone the repository
git clone https://github.com/IINemo/thinkbooster.git
cd thinkbooster

# Create and activate conda environment
conda create -n thinkbooster python=3.11 -y
conda activate thinkbooster

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
- Installs latex2sympy2 for math evaluation (`--no-deps` avoids antlr4 conflict with Hydra)
- Installs vLLM for fast local model inference
- Sets up pre-commit hooks (black, isort, flake8)

---

## Step 3: Configure API Keys

```bash
cp .env.example .env
# Edit .env and add your keys:
# OPENROUTER_API_KEY=sk-or-v1-...
# DEEPSEEK_API_KEY=sk-...
```

**Required for:**
- Running experiments with API models (OPENROUTER_API_KEY)
- Evaluation with LLM judge (DEEPSEEK_API_KEY or OPENROUTER_API_KEY)

---

## Step 4: Verify Installation

```bash
# Run tests to verify setup
pytest tests/strategy_registry.py --validate  # Validate registry
pytest tests/deepconf/ -v                     # Run DeepConf tests
pytest tests/online_best_of_n/ -v             # Run Best-of-N tests

# Or run all tests
make test
```

Expected result: all tests pass (some may skip if API keys are not set).

---

## Step 5: Run Your First Experiment

```bash
# Quick test with 1 sample
python scripts/run_tts_eval.py \
  --config-name experiments/deepconf/run_gsm8k_deepconf_offline \
  dataset.subset=1 \
  strategy.budget=4

# Check the output
ls outputs/  # Results saved here with timestamp
```

---

## Step 6: Make Your First Change

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

See [Strategy Registration Guide](STRATEGY_REGISTRATION.md) for detailed steps.

---

## Step 7: Daily Development Workflow

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

### Pre-commit Hooks

Pre-commit hooks run automatically on `git commit` and will:

- Format code with black and isort
- Check for trailing whitespace and large files
- Run flake8 linting
- Block commit if checks fail

To install or reinstall hooks:

```bash
make hooks
```

---

## Development Commands

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

---

## Pull Request Guidelines

1. Ensure all tests pass: `make test`
2. Run `make fix` and `make lint` before committing
3. Add tests for new strategies (register in `tests/strategy_registry.py`)
4. Keep PRs focused — one feature or fix per PR
5. Write clear commit messages describing what changed and why
