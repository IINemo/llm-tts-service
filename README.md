<img width="130" height="130" alt="image" src="https://github.com/user-attachments/assets/588610f9-f0e2-4bcc-8a71-aa3ffd6af91e" />


# LLM Test-Time Scaling Service

## Install

```bash
./setup.sh
```

This installs:
- Package dependencies (via pip install -e .)
- lm-polygraph dev branch

Update lm-polygraph later:
```bash
./setup.sh --update
```

## Configuration

**API Keys:**
```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your API keys
# OPENROUTER_API_KEY=your-key-here
# DEEPSEEK_API_KEY=your-key-here
```

The script will automatically load API keys from `.env` file.

## Development

Install dev dependencies and pre-commit hooks:
```bash
pip install -e ".[dev]"
make hooks
```

## Running the REST API Service

### Quick Start (Recommended)

```bash
./start_service_app.sh
```

This automated script handles everything:
- Checks Docker is running
- Creates `.env` if needed
- Validates API keys
- Builds and starts the service
- Waits for health check

### Manual Docker

```bash
export OPENROUTER_API_KEY="your-key"
docker-compose up -d
```

### Local Development

For development without Docker:
```bash
pip install -e ".[service]"
export OPENROUTER_API_KEY="your-key"
python service_app/main.py
```

See `service_app/README.md` for detailed service documentation.

**Daily workflow:**
```bash
# Make your changes...

# Before committing - auto-fix all issues
make fix     # Runs pre-commit on all files (black, isort, etc.)

# Check for remaining issues
make lint    # Run flake8

# Commit (hooks will run automatically)
git commit -m "your message"
```

**Manual formatting:**
```bash
make format  # Just run black + isort (without other hooks)
make lint    # Check with flake8
```

Pre-commit hooks run automatically on `git commit` and will block commits that fail checks.

## Project Structure

```
llm-tts-service/
├── config/                           # Hydra configuration files
│   ├── experiments/                  # Complete experiment configs
│   │   ├── deepconf/                # DeepConf experiments (offline/online)
│   │   ├── chain_of_thought/        # Chain-of-thought experiments
│   │   └── self_consistency/        # Self-consistency experiments
│   ├── dataset/                      # Dataset configs (gsm8k, math, etc.)
│   ├── model/                        # Model configs (openai, openrouter, hf, etc.)
│   ├── strategy/                     # Strategy-specific parameters
│   ├── scorer/                       # Scorer configs (PRM, uncertainty, etc.)
│   ├── generation/                   # Generation parameters
│   ├── evaluation/                   # Evaluation configs (llm_judge, alignscore)
│   └── system/                       # System settings (device, seed, etc.)
│
├── llm_tts/                          # Main library package
│   ├── strategies/                   # TTS strategy implementations
│   │   ├── deepconf/                # DeepConf strategy (offline/online modes)
│   │   │   ├── strategy.py          # Main strategy implementation
│   │   │   └── utils.py             # Confidence computation utilities
│   │   ├── strategy_base.py         # Abstract base class for all strategies
│   │   ├── strategy_online_best_of_n.py
│   │   ├── strategy_self_consistency.py
│   │   └── strategy_chain_of_thought.py
│   │
│   ├── models/                       # Model wrappers
│   │   ├── blackboxmodel_with_streaming.py  # OpenAI-compatible model with streaming
│   │   └── base.py                  # Base model interface
│   │
│   ├── scorers/                      # Step scoring implementations
│   │   ├── step_scorer_base.py      # Base scorer interface
│   │   ├── step_scorer_prm.py       # Process Reward Model scorer
│   │   ├── step_scorer_uncertainty.py # Uncertainty-based scorer
│   │   └── majority_voting.py       # Majority voting scorer
│   │
│   ├── evaluation/                   # Evaluation methods
│   │   ├── llm_as_a_judge.py       # LLM-based correctness verification
│   │   ├── exact_match.py          # Direct answer comparison
│   │   └── alignscore.py           # Semantic similarity evaluation
│   │
│   ├── datasets/                     # Dataset utilities
│   │   └── gsm8k.py                # GSM8K dataset loading and processing
│   │
│   ├── early_stopping.py            # Early stopping conditions for streaming
│   ├── step_boundary_detector.py    # Detects step/answer boundaries
│   ├── step_candidate_generator_base.py
│   ├── step_candidate_generator_through_api.py
│   └── step_candidate_generator_through_huggingface.py
│
├── scripts/
│   └── run_tts_eval.py              # Main evaluation script
│
├── tests/                            # Test suite
│   ├── strategy_registry.py         # Strategy registry and validation
│   ├── deepconf/                    # DeepConf strategy tests
│   │   ├── test_deepconf_accurate.py
│   │   ├── test_online_mode.py
│   │   └── test_deepconf_math.py
│   ├── online_best_of_n/            # Best-of-N strategy tests
│   └── run_tts_eval/                # Integration tests
│
├── docs/                             # Documentation
│   ├── deepconf/                    # DeepConf strategy guide
│   │   └── DeepConf.md
│   └── datasets/                    # Dataset documentation
│       └── GSM8K/
│
├── lm-polygraph/                     # Submodule: uncertainty estimation library
│
├── Makefile                          # Development commands (format, lint, test)
├── pyproject.toml                    # Package configuration and dependencies
├── setup.py                          # Package setup
├── setup.sh                          # Installation script
└── .github/workflows/                # CI/CD pipelines
    └── test.yml
```

### Key Components

**Strategies** (`llm_tts/strategies/`)
- `deepconf/` - Confidence-based test-time scaling with offline/online modes
- `strategy_online_best_of_n.py` - Step-by-step generation with PRM scoring
- `strategy_self_consistency.py` - Majority voting across reasoning paths
- `strategy_chain_of_thought.py` - Single-pass step-by-step reasoning

**Models** (`llm_tts/models/`)
- `blackboxmodel_with_streaming.py` - Unified streaming model with early stopping support

**Evaluation** (`scripts/run_tts_eval.py`)
- Two-phase pipeline: generation → evaluation
- Multi-evaluator support (LLM Judge, Exact Match, AlignScore)
- Resume support for long-running experiments

**Configuration** (`config/`)
- Hierarchical Hydra configs with composition
- Pre-configured experiments in `experiments/`
- See `config/README.md` for detailed guide

**Testing & Quality Assurance** (`tests/strategy_registry.py`)
- Centralized registry of all TTS strategies
- Validates each strategy has required tests before merge
- Runs automatically in CI/CD pipeline
- See [Strategy Registration Guide](docs/STRATEGY_REGISTRATION.md) for details

# TODO:
1. Add new scorers
2. Add tree of thought


## Running Experiments

See strategy-specific documentation:
- [DeepConf Strategy](docs/deepconf/DeepConf.md) - Confidence-based test-time scaling

## Examples

**Note:** API keys are loaded from `.env` file - no need to specify them in the command.

### Online Best-of-N Strategy

Reasoner with Qwen-3 (local):
```bash
WANDB_ENTITY=nlpresearch.group WANDB_PROJECT=tts python scripts/run_tts_eval.py \
  --config-path ../config \
  --config-name run_tts_eval \
  dataset=small_gsm8k \
  dataset.subset=1 \
  model=hf_qwen3
```

Reasoner with ChatGPT:
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

### DeepConf Strategy

**Offline mode** (generate N traces, filter by confidence, majority vote):
```bash
WANDB_ENTITY=nlpresearch.group WANDB_PROJECT=tts python scripts/run_tts_eval.py \
  --config-path ../config \
  --config-name experiments/deepconf/run_gsm8k_deepconf_offline \
  model.model_path="openai/gpt-3.5-turbo" \
  dataset.subset=10
```

**Online mode** (adaptive generation with confidence-based early stopping):
```bash
WANDB_ENTITY=nlpresearch.group WANDB_PROJECT=tts python scripts/run_tts_eval.py \
  --config-path ../config \
  --config-name experiments/deepconf/run_gsm8k_deepconf_online \
  model.model_path="openai/gpt-3.5-turbo" \
  dataset.subset=10
```


