.PHONY: help install update clean lint format fix hooks

help:
	@echo "Available commands:"
	@echo "  make install  - Install package and lm-polygraph dev"
	@echo "  make update   - Update lm-polygraph to latest dev"
	@echo "  make hooks    - Install pre-commit hooks"
	@echo "  make fix      - Auto-fix all issues (format + run hooks)"
	@echo "  make format   - Format code (black + isort)"
	@echo "  make lint     - Run linters (flake8)"
	@echo "  make clean    - Remove build artifacts"

install:
	@./setup.sh

update:
	@./setup.sh --update

hooks:
	@echo "Installing pre-commit hooks..."
	@pip install pre-commit
	@pre-commit install
	@echo "✓ Hooks installed. They will run on git commit."

lint:
	@echo "Running flake8..."
	@flake8 llm_tts scripts

format:
	@echo "Formatting with black..."
	@black llm_tts scripts
	@echo "Sorting imports with isort..."
	@isort llm_tts scripts
	@echo "✓ Code formatted"

fix:
	@echo "Auto-fixing all issues..."
	@pre-commit run --all-files || true
	@echo "✓ Done! Check 'make lint' for remaining issues."

test:
	pytest tests/ -v

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

lint:
	flake8 llm_tts scripts
	mypy llm_tts scripts
	black --check llm_tts scripts
	isort --check llm_tts scripts

format:
	black llm_tts scripts
	isort llm_tts scripts

# GSM8K benchmark shortcuts
.PHONY: eval-gsm8k eval-gsm8k-subset

eval-gsm8k:
	@echo "Running GSM8K evaluation with DeepConf..."
	python scripts/run_tts_eval.py \
		--config-path ../config \
		--config-name experiments/run_gsm8k_deepconf

eval-gsm8k-subset:
	@echo "Running GSM8K evaluation on subset (3 samples)..."
	python scripts/run_tts_eval.py \
		--config-path ../config \
		--config-name experiments/run_gsm8k_deepconf \
		dataset.subset=3 \
		strategy.budget=4
