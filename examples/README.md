# LangGraph Uncertainty Routing Example

## Prerequisites

- Docker installed
- `OPENROUTER_API_KEY` set in `.env` file

## Quick Start

### 1. Start the TTS service

```bash
cd service_app
docker compose up --build -d
```

Verify it's running:
```bash
curl http://localhost:8001/health
```

### 2. Run the example

```bash
python examples/langgraph_uncertainty_routing.py
```

### Expected Output

```
============================================================
LangGraph Uncertainty-Aware Routing Demo
============================================================

Question: What is 7 * 8?
----------------------------------------
Status: ACCEPTED
Answer: 56
Confidence: 1.00
Uncertainty: 0.00
Attempts: 1
Final budget: 3
```

## How It Works

1. **ChatTTS** sends request to TTS service with `num_paths=3`
2. Service generates 3 reasoning paths using self-consistency
3. Returns `consensus_score` (confidence) and `uncertainty_score`
4. LangGraph routes based on uncertainty:
   - Low uncertainty → accept answer
   - High uncertainty → retry with higher budget
   - Max attempts reached → escalate

## Configuration

Edit `CONFIG` in `langgraph_uncertainty_routing.py`:

```python
CONFIG = {
    "service_url": "http://localhost:8001/v1",
    "model": "openai/gpt-4o-mini",
    "initial_budget": 3,
    "max_budget": 12,
    "uncertainty_threshold": 0.3,
    "max_attempts": 3,
}
```
