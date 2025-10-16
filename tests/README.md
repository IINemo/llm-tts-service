# Testing Guide for LLM Test-Time Scaling Strategies

This directory contains tests for all TTS strategies. This guide explains what to test, how to structure tests, and best practices.

---

## Table of Contents

1. [Testing Philosophy](#testing-philosophy)
2. [Test Structure](#test-structure)
3. [Strategy Registry](#strategy-registry)
4. [What to Test](#what-to-test)
5. [How to Test New Strategies](#how-to-test-new-strategies)
6. [Running Tests](#running-tests)
7. [CI/CD Integration](#cicd-integration)
8. [Best Practices](#best-practices)

---

## Testing Philosophy

### Core Principles

1. **Isolation**: Test strategy logic independently from infrastructure
2. **Coverage**: Test all critical paths (success, failure, edge cases)
3. **Validation**: Verify both implementation correctness AND output quality
4. **Automation**: All tests should run automatically in CI/CD

### Test Pyramid

```
                    ┌─────────────────┐
                    │  E2E Tests      │  (Smoke tests via run_tts_eval.py)
                    │  - Full pipeline│
                    └─────────────────┘
                           ▲
                ┌──────────────────────┐
                │  Integration Tests   │  (Strategy + Model + API)
                │  - Real API calls    │
                │  - Math validation   │
                └──────────────────────┘
                           ▲
            ┌──────────────────────────────┐
            │  Unit Tests                  │  (Logic validation)
            │  - Answer extraction         │
            │  - Voting logic              │
            │  - Confidence computation    │
            └──────────────────────────────┘
```

---

## Test Structure

### Directory Layout

```
tests/
├── README.md                          # This file
├── deepconf/                          # DeepConf strategy tests
│   ├── test_deepconf_accurate.py      # Unit tests (answer extraction, voting)
│   ├── test_deepconf_math.py          # Integration tests (math problems)
│   └── test_online_mode.py            # Online mode specific tests
├── online_best_of_n/                  # Best-of-N strategy tests
│   └── test_online_best_of_n.py       # Strategy logic tests
├── run_tts_eval/                      # End-to-end pipeline tests
│   └── test_run_tts_eval.py           # Smoke test for full pipeline
└── [your_strategy]/                   # Template for new strategies
    ├── test_[strategy]_logic.py       # Unit tests
    ├── test_[strategy]_integration.py # Integration tests
    └── test_[strategy]_math.py        # Math problem validation
```

### File Naming Convention

- `test_*.py` - PyTest will auto-discover these files
- `test_*_logic.py` - Pure logic tests (no API calls)
- `test_*_integration.py` - Tests with API calls (use `@pytest.mark.skipif`)
- `test_*_math.py` - Math problem validation with expected answers

---

## Strategy Registry

### Overview

**All strategies MUST be registered in `tests/strategy_registry.py`** before they can be merged. This ensures:
- ✅ Every strategy has required tests
- ✅ Tests are validated in CI before merge
- ✅ No strategy is merged without proper testing

### How It Works

1. **Registry File**: `tests/strategy_registry.py` contains `REGISTERED_STRATEGIES` list
2. **CI Validation**: Runs `python tests/strategy_registry.py --validate` before tests
3. **Enforcement**: PR merge is blocked if validation fails

### Registering a New Strategy

When you create a new strategy, you **MUST** add it to the registry:

```python
# In tests/strategy_registry.py, add to REGISTERED_STRATEGIES:

StrategyInfo(
    name="my_strategy",                                    # Strategy name
    class_name="StrategyMyStrategy",                       # Class name
    module_path="llm_tts/strategies/strategy_my_strategy.py",  # Strategy file
    test_dir="tests/my_strategy",                          # Test directory
    required_tests=[
        "test_my_strategy_logic.py",       # Unit tests (required)
        "test_my_strategy_integration.py", # Integration tests (required)
        "test_my_strategy_math.py",        # Math validation (required)
    ],
    description="Brief description of my_strategy",
),
```

### Using the Registry Tool

```bash
# List all registered strategies
python tests/strategy_registry.py --list

# Validate all strategies have required tests
python tests/strategy_registry.py --validate

# Generate template for new strategy
python tests/strategy_registry.py --template my_strategy
```

### Validation Rules

For each registered strategy, the validator checks:

1. ✅ Strategy module exists (`llm_tts/strategies/strategy_*.py`)
2. ✅ Test directory exists (`tests/[strategy_name]/`)
3. ✅ All required test files exist
4. ✅ Test files contain at least one `def test_*()` function

**If ANY check fails, CI will block the PR merge.**

### Example: Adding a New Strategy

```bash
# 1. Generate registry template
python tests/strategy_registry.py --template my_strategy

# 2. Create strategy file
touch llm_tts/strategies/strategy_my_strategy.py

# 3. Create test directory and files
mkdir tests/my_strategy
touch tests/my_strategy/test_my_strategy_logic.py
touch tests/my_strategy/test_my_strategy_integration.py
touch tests/my_strategy/test_my_strategy_math.py

# 4. Add entry to REGISTERED_STRATEGIES (use generated template)
# Edit tests/strategy_registry.py and add the StrategyInfo

# 5. Validate
python tests/strategy_registry.py --validate

# 6. Write actual tests in the test files
# ...

# 7. Commit and push (CI will validate automatically)
git add tests/strategy_registry.py llm_tts/strategies/ tests/my_strategy/
git commit -m "feat: add my_strategy with tests"
```

### What Happens in CI

```yaml
# .github/workflows/test.yml

- name: Validate strategy registry
  run: |
    python tests/strategy_registry.py --validate
    # ❌ Fails if any strategy missing tests
    # ✅ Passes if all strategies have required tests

- name: Test with pytest (all strategies)
  run: |
    pytest tests/ -v
    # Only runs if validation passes
```

### Current Registered Strategies

Run `python tests/strategy_registry.py --list` to see all registered strategies.

As of now:
- ✅ **deepconf** - Confidence-based test-time scaling
- ✅ **online_best_of_n** - Step-by-step with PRM scoring
- ⚠️ **self_consistency** - TODO: Add tests
- ⚠️ **chain_of_thought** - TODO: Add tests

---

## What to Test

### 1. **Strategy Logic** (Unit Tests)

Test the core algorithm logic WITHOUT API calls:

```python
def test_answer_extraction():
    """Test answer extraction from various formats"""
    assert extract_answer("\\boxed{42}") == "42"
    assert extract_answer("answer is \\boxed{x+y}") == "x+y"
    assert extract_answer("no answer here") is None

def test_voting_logic():
    """Test majority voting"""
    votes = [("42", 0.9), ("42", 0.8), ("43", 0.5)]
    winner = majority_vote(votes)
    assert winner == "42"

def test_confidence_computation():
    """Test confidence score calculation"""
    logprobs = [{"logprob": -0.1}, {"logprob": -0.2}]
    confidence = compute_confidence(logprobs)
    assert 0 <= confidence <= 1
```

**What to test:**
- ✅ Answer extraction (all formats)
- ✅ Voting/aggregation logic
- ✅ Confidence/score computation
- ✅ Edge cases (empty input, malformed data)
- ✅ Filtering logic (threshold, top-k)

### 2. **Model Integration** (Integration Tests)

Test strategy with real API calls:

```python
@pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set"
)
def test_strategy_with_model():
    """Test strategy with real model"""
    model = BlackboxModelWithStreaming(
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        model_path="openai/gpt-4o-mini",
        supports_logprobs=True,
        base_url="https://openrouter.ai/api/v1"
    )

    strategy = YourStrategy(model=model, budget=3)
    result = strategy.generate_trajectory("What is 2+2?")

    assert result["completed"] == True
    assert result["trajectory"] != ""
    assert len(result["steps"]) > 0
```

**What to test:**
- ✅ Strategy completes successfully
- ✅ Returns expected data structure
- ✅ Handles API errors gracefully
- ✅ Respects budget/token limits

### 3. **Correctness Validation** (Math Tests)

Test on actual math problems with known answers:

```python
MATH_PROBLEMS = [
    {
        "problem": "What is 15² - 8²? Put your answer in \\boxed{}.",
        "expected": "161",
        "description": "Difference of squares"
    },
    # ... more problems
]

@pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="API key not set"
)
def test_math_accuracy():
    """Test strategy on math problems"""
    model = create_test_model()
    strategy = YourStrategy(model=model)

    correct = 0
    for problem in MATH_PROBLEMS:
        result = strategy.generate_trajectory(problem["problem"])
        answer = extract_answer(result["trajectory"])

        if answer == problem["expected"]:
            correct += 1

    accuracy = correct / len(MATH_PROBLEMS)
    assert accuracy >= 0.6  # Expect at least 60% accuracy
```

**What to test:**
- ✅ Answer correctness on known problems
- ✅ Consistency across runs (temperature=0)
- ✅ Performance vs baseline
- ✅ Edge cases (very hard/easy problems)

### 4. **End-to-End Pipeline** (Smoke Tests)

Test full pipeline via `run_tts_eval.py`:

```python
def test_strategy_in_pipeline():
    """Smoke test: strategy works in full pipeline"""
    cmd = (
        "PYTHONPATH=./ python scripts/run_tts_eval.py "
        "--config-name experiments/your_strategy/config "
        "dataset.subset=1 "
        "model.model_name='openai/gpt-4o-mini'"
    )
    result = subprocess.run(cmd, shell=True, env=test_env)
    assert result.returncode == 0
```

**What to test:**
- ✅ No crashes in full pipeline
- ✅ Config loading works
- ✅ Results are saved correctly
- ✅ Evaluation completes

---

## How to Test New Strategies

### Step 1: Create Test Directory

```bash
mkdir tests/your_strategy
touch tests/your_strategy/test_your_strategy_logic.py
touch tests/your_strategy/test_your_strategy_integration.py
touch tests/your_strategy/test_your_strategy_math.py
```

### Step 2: Write Unit Tests (No API)

```python
# tests/your_strategy/test_your_strategy_logic.py

import pytest
from llm_tts.strategies import YourStrategy
from llm_tts.utils.confidence import extract_answer

def test_answer_extraction():
    """Test answer extraction logic"""
    assert extract_answer("\\boxed{42}") == "42"
    assert extract_answer("The answer is \\boxed{3.14}") == "3.14"

def test_strategy_initialization():
    """Test strategy can be initialized"""
    # Mock model (no API calls)
    mock_model = MockModel()
    strategy = YourStrategy(model=mock_model, budget=5)

    assert strategy.budget == 5
    assert strategy.model == mock_model

def test_filtering_logic():
    """Test trace filtering"""
    traces = [
        {"answer": "42", "confidence": 0.9},
        {"answer": "42", "confidence": 0.8},
        {"answer": "43", "confidence": 0.5},
    ]

    filtered = strategy.filter_traces(traces, top_k=2)
    assert len(filtered) == 2
    assert all(t["confidence"] >= 0.8 for t in filtered)
```

### Step 3: Write Integration Tests (With API)

```python
# tests/your_strategy/test_your_strategy_integration.py

import os
import pytest
from llm_tts.models import BlackboxModelWithStreaming
from llm_tts.strategies import YourStrategy

@pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set"
)
def test_basic_generation():
    """Test strategy generates output"""
    model = BlackboxModelWithStreaming(
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        model_path="openai/gpt-4o-mini",
        supports_logprobs=True,
        base_url="https://openrouter.ai/api/v1"
    )

    strategy = YourStrategy(model=model, budget=3)
    result = strategy.generate_trajectory("What is 5+3?")

    assert result["completed"] == True
    assert "trajectory" in result
    assert "steps" in result
    assert len(result["steps"]) > 0

@pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set"
)
def test_handles_errors():
    """Test strategy handles API errors gracefully"""
    model = BlackboxModelWithStreaming(
        openai_api_key="invalid-key",
        model_path="openai/gpt-4o-mini",
        base_url="https://openrouter.ai/api/v1"
    )

    strategy = YourStrategy(model=model)

    with pytest.raises(Exception):  # Should raise authentication error
        strategy.generate_trajectory("test")
```

### Step 4: Write Math Validation Tests

```python
# tests/your_strategy/test_your_strategy_math.py

import os
import pytest
from llm_tts.models import BlackboxModelWithStreaming
from llm_tts.strategies import YourStrategy
from llm_tts.utils.confidence import extract_answer

MATH_PROBLEMS = [
    {"problem": "Calculate 15² - 8². Put answer in \\boxed{}.", "expected": "161"},
    {"problem": "What is 12 * 8? Put answer in \\boxed{}.", "expected": "96"},
    {"problem": "Sum 1+2+...+10. Put answer in \\boxed{}.", "expected": "55"},
]

@pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set"
)
def test_math_accuracy():
    """Test accuracy on math problems"""
    model = BlackboxModelWithStreaming(
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        model_path="openai/gpt-4o-mini",
        supports_logprobs=True,
        base_url="https://openrouter.ai/api/v1"
    )

    strategy = YourStrategy(model=model, budget=5)

    results = []
    for problem in MATH_PROBLEMS:
        result = strategy.generate_trajectory(problem["problem"])
        answer = extract_answer(result["trajectory"])
        is_correct = (answer == problem["expected"])
        results.append(is_correct)

        print(f"Problem: {problem['problem'][:50]}...")
        print(f"Expected: {problem['expected']}, Got: {answer}")
        print(f"Correct: {is_correct}\n")

    accuracy = sum(results) / len(results)
    print(f"Overall Accuracy: {accuracy:.1%}")

    assert accuracy >= 0.6, f"Accuracy {accuracy:.1%} below threshold"
```

### Step 5: Add Pipeline Integration Test

```python
# tests/run_tts_eval/test_run_tts_eval.py

def test_your_strategy_pipeline():
    """Test your strategy in full pipeline"""
    cmd = (
        "PYTHONPATH=./ python scripts/run_tts_eval.py "
        "--config-path=../config/ "
        "--config-name=experiments/your_strategy/config "
        "dataset.subset=1 "
        "strategy.type=your_strategy"
    )
    exec_result = subprocess.run(cmd, shell=True)
    assert exec_result.returncode == 0, f"Pipeline test failed!"
```

---

## Running Tests

### Run All Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=llm_tts --cov-report=html
```

### Run Specific Test Suites

```bash
# Run only DeepConf tests
pytest tests/deepconf/ -v

# Run only unit tests (fast, no API)
pytest tests/ -k "logic" -v

# Run only integration tests (requires API keys)
pytest tests/ -k "integration" -v

# Run specific test file
pytest tests/deepconf/test_deepconf_math.py -v
```

### Skip API-Dependent Tests

```bash
# Run tests without API keys (skips @pytest.mark.skipif tests)
pytest tests/ -v
# Tests requiring API keys will be SKIPPED automatically
```

### Run with API Keys

```bash
# Set API keys
export OPENROUTER_API_KEY="your-key"
export DEEPSEEK_API_KEY="your-key"

# Run all tests (including API-dependent ones)
pytest tests/ -v
```

---

## CI/CD Integration

### GitHub Actions Workflow

Tests run automatically on:
- ✅ Push to `main` branch
- ✅ Pull requests (opened, synchronized, reopened)
- ✅ Every commit to a PR

### API Keys in CI

API keys are stored as **GitHub Secrets**:

1. Go to: **Repository Settings** → **Secrets and variables** → **Actions**
2. Add secrets:
   - `OPENROUTER_API_KEY`
   - `DEEPSEEK_API_KEY`

The workflow automatically uses these secrets:

```yaml
- name: Test with pytest (all strategies)
  env:
    OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
    DEEPSEEK_API_KEY: ${{ secrets.DEEPSEEK_API_KEY }}
  run: |
    pytest tests/ -v
```

### CI Test Strategy

1. **Fast Unit Tests** - Run first (no API)
2. **Integration Tests** - Run with API keys (if available)
3. **Smoke Tests** - Run full pipeline on subset

---

## Best Practices

### 1. Use `@pytest.mark.skipif` for API Tests

```python
@pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set"
)
def test_with_api():
    # This test is skipped if API key is not set
    pass
```

### 2. Mock External Dependencies

```python
from unittest.mock import Mock, patch

def test_strategy_logic():
    # Mock the model to avoid API calls
    mock_model = Mock()
    mock_model.generate.return_value = {"text": "42", "logprobs": [...]}

    strategy = YourStrategy(model=mock_model)
    result = strategy.generate_trajectory("test")

    assert result["completed"] == True
```

### 3. Test Edge Cases

```python
def test_empty_input():
    """Test strategy handles empty input"""
    strategy = YourStrategy(model=mock_model)
    result = strategy.generate_trajectory("")
    assert result["completed"] == False

def test_malformed_response():
    """Test strategy handles malformed model output"""
    mock_model = Mock()
    mock_model.generate.return_value = {"text": None}  # Malformed!

    strategy = YourStrategy(model=mock_model)
    # Should not crash
    result = strategy.generate_trajectory("test")
```

### 4. Use Fixtures for Common Setup

```python
import pytest

@pytest.fixture
def test_model():
    """Create test model with API key"""
    return BlackboxModelWithStreaming(
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        model_path="openai/gpt-4o-mini",
        supports_logprobs=True,
        base_url="https://openrouter.ai/api/v1"
    )

@pytest.fixture
def math_problems():
    """Load math test problems"""
    return [
        {"problem": "...", "expected": "42"},
        # ...
    ]

def test_with_fixtures(test_model, math_problems):
    """Use fixtures in tests"""
    strategy = YourStrategy(model=test_model)
    # ... test logic
```

### 5. Add Descriptive Assertions

```python
# Bad
assert result["accuracy"] > 0.5

# Good
assert result["accuracy"] > 0.5, (
    f"Accuracy {result['accuracy']:.1%} is below 50% threshold. "
    f"Expected at least 50% on {len(problems)} problems."
)
```

### 6. Test Error Handling

```python
def test_handles_timeout():
    """Test strategy handles timeout gracefully"""
    strategy = YourStrategy(model=slow_model, timeout=1)

    with pytest.raises(TimeoutError):
        strategy.generate_trajectory("complex problem")

def test_handles_rate_limit():
    """Test strategy handles rate limiting"""
    # Mock rate limit error
    mock_model = Mock()
    mock_model.generate.side_effect = RateLimitError()

    strategy = YourStrategy(model=mock_model)

    with pytest.raises(RateLimitError):
        strategy.generate_trajectory("test")
```

---

## Testing Checklist for New Strategies

When implementing a new strategy, ensure you have:

- [ ] **Unit tests** for all core logic (no API calls)
  - [ ] Answer extraction
  - [ ] Voting/aggregation logic
  - [ ] Filtering logic
  - [ ] Score computation
  - [ ] Edge cases

- [ ] **Integration tests** with real model (use `@pytest.mark.skipif`)
  - [ ] Basic generation works
  - [ ] Returns correct data structure
  - [ ] Handles API errors

- [ ] **Math validation tests**
  - [ ] Test on 5+ known problems
  - [ ] Verify answer correctness
  - [ ] Check accuracy threshold

- [ ] **Pipeline integration**
  - [ ] Add test to `test_run_tts_eval.py`
  - [ ] Verify config loading works
  - [ ] Check results are saved

- [ ] **Documentation**
  - [ ] Add docstrings to test functions
  - [ ] Document expected behavior
  - [ ] Add troubleshooting notes

---

## Troubleshooting

### "Tests are skipped in CI"

**Cause:** API keys not configured in GitHub Secrets.

**Solution:**
1. Go to **Repository Settings** → **Secrets and variables** → **Actions**
2. Add `OPENROUTER_API_KEY` and `DEEPSEEK_API_KEY`
3. Re-run workflow

### "Import errors in tests"

**Cause:** Python path not set correctly.

**Solution:**
```bash
# Run from project root
cd /path/to/llm-tts-service
PYTHONPATH=. pytest tests/ -v
```

### "Tests pass locally but fail in CI"

**Cause:** Different environment (dependencies, API keys, etc.)

**Solution:**
1. Check CI logs for specific error
2. Ensure all dependencies in `pyproject.toml`
3. Verify API keys are set in GitHub Secrets
4. Test locally with same Python version as CI

### "API tests are too slow"

**Cause:** Making too many API calls.

**Solution:**
1. Use smaller `dataset.subset` in tests
2. Reduce `budget` in strategy configs
3. Mock API calls in unit tests
4. Use fixtures to reuse model instances

---

## Examples

See these well-tested strategies for reference:

- **DeepConf**: `tests/deepconf/` - Comprehensive test suite with unit, integration, and math tests
- **Online Best-of-N**: `tests/online_best_of_n/` - Integration tests with step-by-step validation

---

## Contributing

When adding tests:

1. Follow the directory structure (`tests/[strategy_name]/`)
2. Use descriptive test names (`test_strategy_handles_empty_input`)
3. Add `@pytest.mark.skipif` for API-dependent tests
4. Document expected behavior in docstrings
5. Run `make test` before committing
6. Ensure CI passes on your PR

---

## References

- **PyTest Documentation**: https://docs.pytest.org/
- **Testing Best Practices**: https://testdriven.io/blog/testing-best-practices/
- **Mocking Guide**: https://docs.python.org/3/library/unittest.mock.html
