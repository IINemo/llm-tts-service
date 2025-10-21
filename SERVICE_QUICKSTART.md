# LLM Test-Time Scaling Service - Quick Start Guide

## Overview

This service provides an OpenAI-compatible REST API for Test-Time Scaling strategies, specifically the DeepConf method for confidence-based reasoning trace filtering and majority voting.

**Service URL**: `http://localhost:8001`
**API Documentation**: `http://localhost:8001/docs`
**Health Check**: `http://localhost:8001/health`

---

## Prerequisites

- Docker and Docker Compose installed
- OpenRouter API key (obtain from https://openrouter.ai/keys)
- `.env` file configured with API key

---

## Installation & Setup

### Step 1: Configure Environment

Create a `.env` file in the repository root:

```bash
cp .env.example .env
```

Edit `.env` and set your API key:

```bash
OPENROUTER_API_KEY=sk-your-actual-key-here
```

### Step 2: Start the Service

**Option 1: Automated Script (Recommended)**
```bash
./start_service_app.sh
```

**Option 2: Manual Docker**
```bash
docker-compose up -d
```

### Step 3: Verify Service is Running

```bash
# Check health
curl http://localhost:8001/health

# Expected response:
# {"status":"healthy","version":"1.0.0"}
```

---

## Using the DeepConf Strategy

### Basic Request Structure

```bash
curl -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-4o-mini",
    "messages": [
      {"role": "user", "content": "Your question here"}
    ],
    "tts_strategy": "deepconf",
    "tts_budget": 8,
    "tts_mode": "offline",
    "tts_filter_method": "top5"
  }'
```

### Request Parameters

**Standard OpenAI Parameters:**
- `model` (string, required): Model identifier
  - Recommended: `"openai/gpt-4o-mini"` (fast, cost-effective)
  - Alternatives: `"openai/gpt-4o"`, `"openai/gpt-3.5-turbo"`
- `messages` (array, required): Chat messages in OpenAI format
  ```json
  [{"role": "user", "content": "Question text"}]
  ```

**DeepConf-Specific Parameters:**
- `tts_strategy` (string): Set to `"deepconf"`
- `tts_budget` (integer): Number of reasoning traces to generate (4-16 recommended)
- `tts_mode` (string): `"offline"` (default) or `"online"`
- `tts_filter_method` (string): `"top3"`, `"top5"`, or `"top10"`

### Response Format

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "model": "openai/gpt-4o-mini",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "Step-by-step solution text..."
    },
    "finish_reason": "stop",
    "tts_metadata": {
      "strategy": "deepconf",
      "elapsed_time": 17.5,
      "extracted_answer": "42",
      "completed": true
    }
  }],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 150,
    "total_tokens": 160
  }
}
```

---

## Working Example

### Request

```bash
curl -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-4o-mini",
    "messages": [
      {
        "role": "user",
        "content": "Solve this step-by-step and provide the final answer in \\boxed{}: Calculate 8 times 9."
      }
    ],
    "tts_strategy": "deepconf",
    "tts_budget": 6,
    "tts_mode": "offline",
    "tts_filter_method": "top5"
  }'
```

### Expected Output

- **Extracted Answer**: 72
- **Processing Time**: ~30-40 seconds (depending on trace complexity)
- **Traces Generated**: 6 reasoning traces
- **Filtering**: Top 5 traces by confidence used for majority voting

---

## Python Client Example

### Using OpenAI SDK

```python
from openai import OpenAI

# Initialize client pointing to local service
client = OpenAI(
    base_url="http://localhost:8001/v1",
    api_key="dummy"  # Not required for local service
)

# Make request
response = client.chat.completions.create(
    model="openai/gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "Solve step-by-step: What is 15 * 23?"
        }
    ],
    extra_body={
        "tts_strategy": "deepconf",
        "tts_budget": 8,
        "tts_mode": "offline",
        "tts_filter_method": "top5"
    }
)

# Extract results
answer = response.choices[0].message.content
metadata = response.choices[0].tts_metadata

print(f"Answer: {answer}")
print(f"Extracted: {metadata['extracted_answer']}")
print(f"Time: {metadata['elapsed_time']:.2f}s")
```

---

## Service Management

### View Logs
```bash
docker-compose logs -f tts-service
```

### Stop Service
```bash
docker-compose down
```

### Restart Service
```bash
docker-compose restart tts-service
```

### Check Running Containers
```bash
docker-compose ps
```

---

## Troubleshooting

### Service Not Starting

**Check Docker is running:**
```bash
docker info
```

**Verify API key is set:**
```bash
grep OPENROUTER_API_KEY .env
```

**Check logs for errors:**
```bash
docker-compose logs tts-service
```

### Empty or Invalid Responses

**Issue**: Model returns very short responses without extractable answers.

**Solution**: Ensure your prompt explicitly requests step-by-step reasoning and uses `\boxed{}` format:

```json
{
  "messages": [{
    "role": "user",
    "content": "Solve this step-by-step and provide the final answer in \\boxed{}: [question]"
  }]
}
```

### Slow Response Times

- Default processing time: 20-60 seconds depending on trace budget
- Reduce `tts_budget` for faster responses (minimum: 4)
- Use `openai/gpt-4o-mini` instead of `openai/gpt-4o` for better performance

---

## API Endpoints Reference

### POST /v1/chat/completions
Create a chat completion with DeepConf test-time scaling.

### GET /v1/models
List available models.

**Example:**
```bash
curl http://localhost:8001/v1/models
```

**Response:**
```json
{
  "object": "list",
  "data": [
    {"id": "openai/gpt-4o-mini", "object": "model"},
    {"id": "openai/gpt-4o", "object": "model"},
    {"id": "openai/gpt-3.5-turbo", "object": "model"}
  ]
}
```

### GET /health
Service health check.

---

## Configuration Options

### DeepConf Strategy Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tts_budget` | integer | 8 | Number of reasoning traces to generate |
| `tts_mode` | string | "offline" | Mode: "offline" or "online" |
| `tts_filter_method` | string | "top5" | Filtering: "top3", "top5", "top10" |
| `temperature` | float | 0.7 | Sampling temperature for diversity |
| `max_tokens` | integer | 4096 | Maximum tokens per trace |

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENROUTER_API_KEY` | Yes | OpenRouter API key for model access |
| `PORT` | No | Service port (default: 8001) |
| `HOST` | No | Service host (default: 0.0.0.0) |

---

## Production Considerations

1. **API Key Security**: Never commit `.env` file to version control
2. **Rate Limiting**: Consider implementing rate limits for production deployments
3. **Monitoring**: Use `/health` endpoint for uptime monitoring
4. **Logging**: Service logs to stdout; configure Docker logging drivers as needed
5. **Scaling**: Current implementation uses in-memory model caching; consider external cache for multi-instance deployments

---

## Current Limitations

1. **Streaming**: Not currently supported (responses are buffered)
2. **Available Strategies**: Only DeepConf is implemented
   - Self-Consistency: Planned
   - Best-of-N: Planned
3. **Authentication**: Not required in current version

---

## Support

For issues or questions, please refer to the main repository documentation or contact the development team.

**Version**: 1.0.0
**Last Updated**: October 2025
