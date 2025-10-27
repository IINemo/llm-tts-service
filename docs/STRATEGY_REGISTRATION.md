# Strategy Registration System

This document explains how the strategy registration system enforces test coverage for all TTS strategies.

---

## Overview

**Problem**: Developers could previously merge new strategies without proper tests, leading to untested code in production.

**Solution**: A registry system that:
1. ✅ Tracks all implemented strategies
2. ✅ Validates each strategy has required tests
3. ✅ Blocks PR merges if tests are missing
4. ✅ Runs automatically in CI/CD pipeline

---

## How It Works

### 1. Strategy Registry (`tests/strategy_registry.py`)

Central registry of all strategies with metadata:

```python
REGISTERED_STRATEGIES = [
    StrategyInfo(
        name="deepconf",
        class_name="StrategyDeepConf",
        module_path="llm_tts/strategies/strategy_deepconf.py",
        test_dir="tests/deepconf",
        required_tests=[
            "test_deepconf_accurate.py",
            "test_online_mode.py",
        ],
        description="Confidence-based test-time scaling",
    ),
    # ... more strategies
]
```

### 2. Validation Rules

For each registered strategy, the validator checks:

| Check | Description | Blocks Merge |
|-------|-------------|--------------|
| Strategy module exists | `llm_tts/strategies/strategy_*.py` must exist | ✅ Yes |
| Test directory exists | `tests/[strategy_name]/` must exist | ✅ Yes |
| Required test files exist | All files in `required_tests` must exist | ✅ Yes |
| Test files have tests | Must contain at least one `def test_*()` | ✅ Yes |

### 3. CI/CD Integration

```yaml
# .github/workflows/test.yml

- name: Validate strategy registry
  run: |
    python tests/strategy_registry.py --validate
    # ❌ Exit code 1 if validation fails (blocks merge)
    # ✅ Exit code 0 if all strategies valid

- name: Test with pytest
  run: |
    pytest tests/ -v
    # Only runs if validation passes
```

**CI Process:**
1. Developer pushes code or opens PR
2. CI runs `strategy_registry.py --validate`
3. If validation fails → CI fails → PR cannot merge
4. If validation passes → Run pytest → Merge allowed if tests pass

---

## Adding a New Strategy

### Step-by-Step Guide

#### 1. Generate Registry Template

```bash
python tests/strategy_registry.py --template my_strategy
```

Output:
```python
StrategyInfo(
    name="my_strategy",
    class_name="StrategyMyStrategy",
    module_path="llm_tts/strategies/strategy_my_strategy.py",
    test_dir="tests/my_strategy",
    required_tests=[
        "test_my_strategy_logic.py",
        "test_my_strategy_integration.py",
        "test_my_strategy_math.py",
    ],
    description="TODO: Brief description",
),
```

#### 2. Create Strategy Implementation

```bash
# Create strategy file
touch llm_tts/strategies/strategy_my_strategy.py
```

Implement your strategy:
```python
# llm_tts/strategies/strategy_my_strategy.py

from .strategy_base import StrategyBase

class StrategyMyStrategy(StrategyBase):
    def __init__(self, model, **kwargs):
        self.model = model
        # ... initialization

    def generate_trajectory(self, prompt):
        # ... implementation
        return {
            "trajectory": "...",
            "steps": [...],
            "validity_scores": [...],
            "completed": True,
            "metadata": {...}
        }
```

#### 3. Create Test Structure

```bash
# Create test directory
mkdir tests/my_strategy

# Create required test files
touch tests/my_strategy/test_my_strategy_logic.py
touch tests/my_strategy/test_my_strategy_integration.py
touch tests/my_strategy/test_my_strategy_math.py
```

#### 4. Write Tests

**Unit Tests** (no API calls):
```python
# tests/my_strategy/test_my_strategy_logic.py

def test_answer_extraction():
    """Test answer extraction logic"""
    from llm_tts.utils.confidence import extract_answer
    assert extract_answer("\\boxed{42}") == "42"

def test_strategy_initialization():
    """Test strategy can be initialized"""
    from llm_tts.strategies import StrategyMyStrategy
    from unittest.mock import Mock

    mock_model = Mock()
    strategy = StrategyMyStrategy(model=mock_model)
    assert strategy.model == mock_model
```

**Integration Tests** (with API):
```python
# tests/my_strategy/test_my_strategy_integration.py

import os
import pytest
from llm_tts.models import BlackboxModelWithStreaming
from llm_tts.strategies import StrategyMyStrategy

@pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set"
)
def test_strategy_with_real_model():
    """Test strategy with real API"""
    model = BlackboxModelWithStreaming(
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        model_path="openai/gpt-4o-mini",
        supports_logprobs=True,
        base_url="https://openrouter.ai/api/v1"
    )

    strategy = StrategyMyStrategy(model=model)
    result = strategy.generate_trajectory("What is 2+2?")

    assert result["completed"] == True
    assert len(result["steps"]) > 0
```

**Math Validation**:
```python
# tests/my_strategy/test_my_strategy_math.py

import os
import pytest
from llm_tts.models import BlackboxModelWithStreaming
from llm_tts.strategies import StrategyMyStrategy
from llm_tts.utils.confidence import extract_answer

MATH_PROBLEMS = [
    {"problem": "15² - 8²? Answer in \\boxed{}", "expected": "161"},
    {"problem": "12 * 8? Answer in \\boxed{}", "expected": "96"},
]

@pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="API key not set"
)
def test_math_accuracy():
    """Test accuracy on known problems"""
    model = BlackboxModelWithStreaming(...)
    strategy = StrategyMyStrategy(model=model)

    correct = 0
    for problem in MATH_PROBLEMS:
        result = strategy.generate_trajectory(problem["problem"])
        answer = extract_answer(result["trajectory"])
        if answer == problem["expected"]:
            correct += 1

    accuracy = correct / len(MATH_PROBLEMS)
    assert accuracy >= 0.5, f"Accuracy {accuracy:.1%} too low"
```

#### 5. Register the Strategy

Edit `tests/strategy_registry.py` and add to `REGISTERED_STRATEGIES`:

```python
REGISTERED_STRATEGIES = [
    # ... existing strategies

    StrategyInfo(
        name="my_strategy",
        class_name="StrategyMyStrategy",
        module_path="llm_tts/strategies/strategy_my_strategy.py",
        test_dir="tests/my_strategy",
        required_tests=[
            "test_my_strategy_logic.py",
            "test_my_strategy_integration.py",
            "test_my_strategy_math.py",
        ],
        description="My custom test-time scaling strategy",
    ),
]
```

#### 6. Validate Locally

```bash
# Run validation
python tests/strategy_registry.py --validate

# Expected output:
# ✅ ALL STRATEGIES VALID

# Run tests
pytest tests/my_strategy/ -v
```

#### 7. Commit and Push

```bash
git add llm_tts/strategies/strategy_my_strategy.py
git add tests/my_strategy/
git add tests/strategy_registry.py

git commit -m "feat: add my_strategy with comprehensive tests"
git push
```

CI will automatically:
1. Validate strategy is registered
2. Verify all required tests exist
3. Run pytest on all tests
4. Block merge if any step fails

---

## Registry Commands

### List All Strategies

```bash
python tests/strategy_registry.py --list
```

Output:
```
REGISTERED TTS STRATEGIES

Total strategies: 2

1. deepconf
   Class: StrategyDeepConf
   Description: Confidence-based test-time scaling with trace filtering
   Module: llm_tts/strategies/strategy_deepconf.py
   Test dir: tests/deepconf
   Required tests:
     - test_deepconf_accurate.py
     - test_online_mode.py

2. online_best_of_n
   ...
```

### Validate All Strategies

```bash
python tests/strategy_registry.py --validate
```

Success output:
```
✅ ALL STRATEGIES VALID
```

Failure output:
```
❌ VALIDATION FAILED

Fix these issues before merging:
  • my_strategy: Test file missing: test_my_strategy_logic.py
  • my_strategy: Test directory not found: tests/my_strategy
```

### Generate Template

```bash
python tests/strategy_registry.py --template beam_search
```

Output:
```python
StrategyInfo(
    name="beam_search",
    class_name="StrategyBeamSearch",
    module_path="llm_tts/strategies/strategy_beam_search.py",
    test_dir="tests/beam_search",
    required_tests=[
        "test_beam_search_logic.py",
        "test_beam_search_integration.py",
        "test_beam_search_math.py",
    ],
    description="TODO: Brief description of beam_search strategy",
),
```

---

## Testing Requirements

### Minimum Required Tests

Every strategy **MUST** have:

1. **Unit Tests** (`test_*_logic.py`)
   - Test core algorithm logic
   - No API calls (use mocks)
   - Fast execution (<1 second)

2. **Integration Tests** (`test_*_integration.py`)
   - Test with real model/API
   - Use `@pytest.mark.skipif` for API key checks
   - Slower execution (API calls)

3. **Math Validation** (`test_*_math.py`)
   - Test on known problems
   - Verify answer correctness
   - Check accuracy threshold

### Optional Tests

- Performance benchmarks
- Stress tests
- Edge case tests
- Comparison with baselines

---

## Current Registered Strategies

| Strategy | Status | Tests |
|----------|--------|-------|
| **deepconf** | ✅ Complete | 2 test files (accurate, online_mode) |
| **online_best_of_n** | ✅ Complete | 1 test file (logic) |
| **self_consistency** | ⚠️ TODO | Not registered (missing tests) |
| **chain_of_thought** | ⚠️ TODO | Not registered (missing tests) |

---

## FAQ

### Q: What happens if I don't register my strategy?

**A:** Your strategy won't be validated in CI, but it also won't block other PRs. However, it's against project policy to merge unregistered strategies.

### Q: Can I temporarily skip validation?

**A:** No. The validation is mandatory for all PRs. If you're working on a strategy, create tests alongside the implementation.

### Q: What if my tests require expensive API calls?

**A:** Use `@pytest.mark.skipif` to skip tests when API keys aren't available. Tests will run in CI with secrets configured.

### Q: Can I have fewer than 3 test files?

**A:** Yes, adjust `required_tests` in the registry for your strategy. Minimum: 1 test file with at least one test function.

### Q: How do I test locally without API keys?

**A:** Write unit tests that don't need API keys. Use mocks for the model. Integration tests will be skipped automatically.

---

## Benefits

### For Developers

- ✅ Clear testing expectations
- ✅ Template generation for new strategies
- ✅ Catch missing tests before review
- ✅ No surprise CI failures

### For Reviewers

- ✅ Guaranteed test coverage
- ✅ Consistent test structure
- ✅ Easy to verify completeness
- ✅ Less back-and-forth on PRs

### For the Project

- ✅ High test coverage
- ✅ Reliable CI/CD
- ✅ Easier maintenance
- ✅ Better code quality

---

## References

- **Testing Guide**: `tests/README.md`
- **Registry Implementation**: `tests/strategy_registry.py`
- **CI Workflow**: `.github/workflows/test.yml`
- **Example Tests**: `tests/deepconf/`, `tests/online_best_of_n/`
