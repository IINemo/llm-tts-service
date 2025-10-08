# LLM Test-Time Scaling Service

OpenAI-compatible REST API for Test-Time Scaling strategies.

## Overview

This service exposes TTS strategies (DeepConf, Best-of-N, etc.) through an **OpenAI-compatible API**. You can use it as a drop-in replacement for OpenAI's API, allowing you to leverage advanced test-time scaling with your existing OpenAI SDK code.

## Features

- ✅ **OpenAI-Compatible**: Works with OpenAI Python SDK and any OpenAI-compatible clients
- ✅ **Multiple TTS Strategies**: DeepConf (offline/online), Best-of-N, Self-Consistency
- ✅ **Easy Integration**: Change only the `base_url` in your existing code
- ✅ **Auto-Documentation**: Interactive API docs at `/docs`
- ✅ **Production-Ready**: CORS support, error handling, logging

## Quick Start

```bash
# From repository root
export OPENROUTER_API_KEY="your-key"
docker-compose up -d
```

The service will start on `http://localhost:8001`

Open http://localhost:8001/docs for interactive API documentation.

## Usage

### With OpenAI Python SDK

```python
from openai import OpenAI

# Point to your local TTS service
client = OpenAI(
    base_url="http://localhost:8001/v1",
    api_key="dummy"  # Not required yet
)

# Use exactly like OpenAI's API
response = client.chat.completions.create(
    model="openai/gpt-4o-mini",
    messages=[
        {"role": "user", "content": "Solve step by step: 2+2=?"}
    ],
    # TTS-specific parameters (optional)
    extra_body={
        "tts_strategy": "deepconf",
        "tts_mode": "offline",
        "tts_budget": 8,
        "tts_filter_method": "top5"
    }
)

print(response.choices[0].message.content)

# Access TTS metadata
tts_metadata = response.choices[0].tts_metadata
print(f"Confidence: {tts_metadata['confidence']}")
print(f"Agreement: {tts_metadata['agreement']}")
```

### With cURL

```bash
curl -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-4o-mini",
    "messages": [
      {"role": "user", "content": "Solve: 2+2=?"}
    ],
    "tts_strategy": "deepconf",
    "tts_budget": 8
  }'
```

## API Endpoints

### POST /v1/chat/completions

Create a chat completion with TTS strategy.

**Standard OpenAI Parameters:**
- `model` (string, required): Model to use (e.g., "openai/gpt-4o-mini")
- `messages` (array, required): Chat messages
- `temperature` (float): Sampling temperature (0-2, default: 0.7)
- `top_p` (float): Nucleus sampling (0-1, default: 1.0)
- `max_tokens` (int): Maximum tokens to generate (default: 512)

**TTS-Specific Parameters:**
- `tts_strategy` (string): Strategy to use ("deepconf", default)
- `tts_mode` (string): DeepConf mode ("offline" or "online", default: "offline")
- `tts_budget` (int): Number of reasoning traces (default: 8)
- `tts_filter_method` (string): Filtering method ("top5", "top10", default: "top5")

**Response:**
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "openai/gpt-4o-mini",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "Step-by-step solution..."
    },
    "finish_reason": "stop",
    "tts_metadata": {
      "strategy": "deepconf",
      "num_traces": 8,
      "confidence": 16.2,
      "agreement": 1.0
    }
  }],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 50,
    "total_tokens": 60
  }
}
```

### GET /v1/models

List available models.

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "openai/gpt-4o-mini",
      "object": "model",
      "created": 1234567890,
      "owned_by": "openai"
    }
  ]
}
```

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

## TTS Strategy Configuration

### DeepConf Offline Mode

Generate multiple reasoning traces, filter by confidence, use majority voting:

```python
response = client.chat.completions.create(
    model="openai/gpt-4o-mini",
    messages=[...],
    extra_body={
        "tts_strategy": "deepconf",
        "tts_mode": "offline",
        "tts_budget": 8,           # Generate 8 traces
        "tts_filter_method": "top5" # Use top-5 for voting
    }
)
```

### DeepConf Online Mode

Adaptive generation with confidence-based early stopping:

```python
response = client.chat.completions.create(
    model="openai/gpt-4o-mini",
    messages=[...],
    extra_body={
        "tts_strategy": "deepconf",
        "tts_mode": "online",
        "tts_budget": 10,  # Total budget (warmup + adaptive)
    }
)
```

## Configuration

### Environment Variables

Create a `.env` file in the `service/` directory:

```bash
# API Keys
OPENROUTER_API_KEY=your-key-here
OPENAI_API_KEY=your-key-here

# Server Settings
HOST=0.0.0.0
PORT=8000

# DeepConf Defaults
DEEPCONF_BUDGET=8
DEEPCONF_FILTER_METHOD=top5
DEEPCONF_TEMPERATURE=0.7
```

### Settings

All settings can be configured via environment variables or in `service/core/config.py`:

- `API_TITLE`: Service title
- `API_VERSION`: API version
- `HOST`: Server host (default: "0.0.0.0")
- `PORT`: Server port (default: 8000)
- `DEFAULT_MODEL`: Default model (default: "openai/gpt-4o-mini")
- `DEFAULT_STRATEGY`: Default TTS strategy (default: "deepconf")

## Docker Deployment

### Quick Start with Docker Compose (Recommended)

```bash
# From repository root
docker-compose up -d

# View logs
docker-compose logs -f tts-service

# Stop
docker-compose down
```

Make sure you have `OPENROUTER_API_KEY` in your environment or create a `.env` file:

```bash
# .env file
OPENROUTER_API_KEY=your-key-here
DEEPCONF_BUDGET=8
DEEPCONF_FILTER_METHOD=top5
```

### Manual Docker Build

```bash
# Build image
docker build -f service/Dockerfile -t llm-tts-service .

# Run container
docker run -d \
  --name tts-service \
  -p 8001:8001 \
  -e OPENROUTER_API_KEY="your-key" \
  llm-tts-service

# View logs
docker logs -f tts-service
```

The service will be available at **http://localhost:8001**

## Development

### Adding New TTS Strategies

To add a new TTS method to the service:

**1. Implement the strategy** in `llm_tts/strategies/`:

```python
# llm_tts/strategies/strategy_your_method.py
from llm_tts.strategies.strategy_base import StrategyBase

class StrategyYourMethod(StrategyBase):
    def __init__(self, model, budget=8, **kwargs):
        self.model = model
        self.budget = budget

    def generate_trajectory(self, prompt):
        # Your implementation
        return {
            "trajectory": "generated text",
            "steps": [...],
            "completed": True,
            "metadata": {...}
        }
```

**2. Add to strategy manager** in `service/core/strategy_manager.py`:

```python
# Import your strategy
from llm_tts.strategies.strategy_your_method import StrategyYourMethod

# In StrategyManager class:
def create_strategy(self, strategy_type, model_name, strategy_config):
    if strategy_type == "deepconf":
        return self._create_deepconf_strategy(model_name, strategy_config)
    elif strategy_type == "your_method":  # Add this
        return self._create_your_method_strategy(model_name, strategy_config)
    # ...

def _create_your_method_strategy(self, model_name, config):
    model = self._get_or_create_model(model_name, ...)
    return StrategyYourMethod(
        model=model,
        budget=config.get("budget", 8),
        # Add your strategy parameters
    )
```

**3. Test your strategy**:

```bash
curl -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-4o-mini",
    "messages": [{"role": "user", "content": "Test question"}],
    "tts_strategy": "your_method",
    "tts_budget": 8
  }'
```

**4. Add tests** in `tests/`:

```python
def test_your_method_strategy():
    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[...],
        extra_body={"tts_strategy": "your_method"}
    )
    assert response.choices[0].message.content
```

### Project Structure

```
service/
├── api/
│   ├── routes/              # API endpoint handlers
│   │   ├── chat.py         # /v1/chat/completions
│   │   └── models.py       # /v1/models
│   └── models/             # Pydantic schemas
│       └── openai_compat.py  # OpenAI-compatible models
├── core/
│   ├── config.py           # Configuration
│   └── strategy_manager.py # TTS strategy management
├── main.py                 # FastAPI app
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## Troubleshooting

### "API key not set"

Make sure you've exported `OPENROUTER_API_KEY`:

```bash
export OPENROUTER_API_KEY="your-key-here"
```

### "Streaming not supported"

Set `stream=False` in your request (streaming will be added in future versions).

### Import errors

Make sure you're running from the repository root:

```bash
# From llm-tts-service/ directory
PYTHONPATH=. python service/main.py
```

## Examples

See `service/examples/` for complete examples:
- `simple_client.py` - Basic usage with OpenAI SDK
- `batch_requests.py` - Processing multiple prompts
- `compare_strategies.py` - Comparing different TTS strategies

## License

Same as main repository.

## Support

For issues and questions, please open an issue on GitHub.
