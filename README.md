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

## Structure
* config -- hydra configuration files.
* llm_tts -- the library with test time scaling strategies.
* scripts/run_tts_eval.py -- the script for running evaluation of test time scaling methods.

# TODO:
1. Add new scorers
2. Add tree of thought


## Running Experiments

See strategy-specific documentation:
- [DeepConf Strategy](docs/deepconf/DeepConf.md) - Confidence-based test-time scaling
