.PHONY: help install update clean

help:
	@echo "Available commands:"
	@echo "  make install  - Install package and lm-polygraph dev"
	@echo "  make update   - Update lm-polygraph to latest dev"
	@echo "  make clean    - Remove build artifacts"

install:
	@./setup.sh

update:
	@./setup.sh --update

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
