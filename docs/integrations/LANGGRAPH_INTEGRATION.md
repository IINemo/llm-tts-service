# LangChain/LangGraph Integration for llm-tts-service

This document describes approaches for integrating llm-tts-service with LangChain and LangGraph, enabling uncertainty-aware LLM agents.

## TL;DR - Which Approach to Use?

| Approach | Use When | Production Ready |
|----------|----------|------------------|
| **ChatTTS** (Custom ChatModel) | You control both client and server | Yes |
| **Logprobs Tunnel** (Hack) | Quick PoC, can't modify client code | No |
| **Direct API** (No LangChain) | Don't need LangChain features | Yes |

**Recommendation**: Use `ChatTTS` for production. It's ~40 lines of code and semantically correct.

---

## Motivation

Current LLM agents return answers without confidence information. By integrating llm-tts-service with LangGraph, we can:

1. **Return uncertainty alongside answers** - Agents know when they're uncertain
2. **Enable uncertainty-based routing** - Retry, escalate, or accept based on confidence
3. **Improve reliability** - Use test-time scaling for critical decisions
4. **Support human-in-the-loop** - Flag uncertain answers for review

---

## Available Uncertainty Metrics

llm-tts-service provides several uncertainty/confidence metrics:

| Metric | Strategy | Range | Description |
|--------|----------|-------|-------------|
| `confidence` | DeepConf | 0-1 | Token-level logprob confidence (higher = more confident) |
| `uncertainty` | lm-polygraph | 0-1 | Predictive distribution uncertainty (lower = more confident) |
| `consensus_ratio` | Self-Consistency | 0-1 | Fraction of traces agreeing on answer |
| `prm_score` | Best-of-N | varies | Process reward model score |
| `num_consistent` | Majority Voting | int | Count of traces with same answer |

---

## Integration Approaches

### Approach 1: LangGraph Tool Wrapper

**Description**: Wrap TTS strategies as LangChain tools that return structured output with uncertainty.

**Architecture**:
```
LangGraph Agent
    │
    ▼
┌─────────────────┐
│  TTS Tool       │ ◄── LangChain @tool decorator
│  (in-process)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  TTS Strategy   │ ◄── DeepConf, Self-Consistency, etc.
│  (llm_tts)      │
└─────────────────┘
```

**Implementation**:

```python
# llm_tts/integrations/langchain_tool.py

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Literal

class TTSResult(BaseModel):
    """Structured result from TTS strategy with uncertainty quantification."""
    answer: str = Field(description="The final extracted answer")
    confidence: float = Field(ge=0, le=1, description="Confidence score (1 = highly confident)")
    uncertainty: float = Field(ge=0, le=1, description="Uncertainty estimate (0 = highly certain)")
    reasoning: str = Field(description="Best reasoning trace / chain of thought")
    num_traces: int = Field(description="Number of reasoning traces generated")
    consensus_ratio: float = Field(ge=0, le=1, description="Agreement ratio among traces")
    strategy_used: str = Field(description="TTS strategy that was used")

@tool
def tts_reasoning(
    question: str,
    strategy: Literal["deepconf", "self_consistency", "best_of_n"] = "deepconf",
    budget: int = 8
) -> TTSResult:
    """
    Solve a reasoning problem using test-time scaling with uncertainty quantification.

    Use this tool when you need:
    - High-confidence answers for critical decisions
    - Uncertainty estimates to know when to escalate
    - Multiple reasoning paths for complex problems

    Args:
        question: The problem or question to solve
        strategy: TTS strategy to use (deepconf recommended for confidence)
        budget: Number of reasoning traces to generate (higher = more reliable)

    Returns:
        TTSResult with answer, confidence, uncertainty, and reasoning trace
    """
    from llm_tts.integrations.langgraph_adapter import run_tts_strategy

    result = run_tts_strategy(
        question=question,
        strategy=strategy,
        budget=budget
    )

    return TTSResult(
        answer=result["answer"],
        confidence=result["confidence"],
        uncertainty=1 - result["confidence"],
        reasoning=result["best_trace"],
        num_traces=result["num_traces"],
        consensus_ratio=result["consensus_ratio"],
        strategy_used=strategy
    )
```

**Pros**:
- Simplest to implement and test
- No network overhead (in-process)
- Full access to all TTS internals
- Easy debugging

**Cons**:
- Tight coupling with LangGraph process
- GPU memory shared with agent
- Not suitable for distributed systems

**Best for**: Research, prototyping, single-machine deployments

---

### Approach 2: LangGraph Node with Conditional Routing

**Description**: Create a LangGraph node that uses uncertainty to control agent flow - retry with higher budget, try different strategy, or escalate to human.

**Architecture**:
```
                    ┌──────────────────┐
                    │   Start Node     │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
              ┌────►│  TTS Reasoning   │◄────┐
              │     │     Node         │     │
              │     └────────┬─────────┘     │
              │              │               │
              │              ▼               │
              │     ┌──────────────────┐     │
              │     │ Check Uncertainty │     │
              │     └────────┬─────────┘     │
              │              │               │
        retry │    ┌─────────┼─────────┐     │ different
        higher│    │         │         │     │ strategy
        budget│    ▼         ▼         ▼     │
              │  accept   escalate   retry───┘
              │    │         │
              │    ▼         ▼
              │   END    Human Review
              │
              └─── (uncertainty > 0.3
                    AND attempts < 3)
```

**Implementation**:

```python
# llm_tts/integrations/langgraph_nodes.py

from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal
from llm_tts.integrations.langchain_tool import tts_reasoning

class AgentState(TypedDict):
    question: str
    answer: str | None
    uncertainty: float
    confidence: float
    reasoning: str | None
    attempts: int
    budget: int
    strategy: str
    status: Literal["pending", "accepted", "escalated"]

def tts_reasoning_node(state: AgentState) -> AgentState:
    """Execute TTS strategy and update state with results."""
    result = tts_reasoning.invoke({
        "question": state["question"],
        "strategy": state["strategy"],
        "budget": state["budget"]
    })

    return {
        **state,
        "answer": result.answer,
        "uncertainty": result.uncertainty,
        "confidence": result.confidence,
        "reasoning": result.reasoning,
        "attempts": state["attempts"] + 1
    }

def increase_budget_node(state: AgentState) -> AgentState:
    """Increase budget for retry."""
    return {
        **state,
        "budget": min(state["budget"] * 2, 32)  # Cap at 32
    }

def try_different_strategy_node(state: AgentState) -> AgentState:
    """Switch to a different TTS strategy."""
    strategy_fallback = {
        "deepconf": "self_consistency",
        "self_consistency": "best_of_n",
        "best_of_n": "deepconf"
    }
    return {
        **state,
        "strategy": strategy_fallback.get(state["strategy"], "deepconf"),
        "budget": 8  # Reset budget for new strategy
    }

def route_by_uncertainty(state: AgentState) -> Literal["accept", "retry_budget", "retry_strategy", "escalate"]:
    """Route based on uncertainty and attempt count."""
    uncertainty = state["uncertainty"]
    attempts = state["attempts"]

    # High confidence - accept
    if uncertainty < 0.2:
        return "accept"

    # Medium uncertainty - retry with more budget (up to 2 retries)
    if uncertainty < 0.4 and attempts < 3:
        return "retry_budget"

    # Still uncertain - try different strategy (up to 4 total attempts)
    if uncertainty < 0.6 and attempts < 5:
        return "retry_strategy"

    # Very uncertain or too many attempts - escalate
    return "escalate"

def build_uncertainty_aware_graph() -> StateGraph:
    """Build LangGraph with uncertainty-based routing."""

    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("tts_reasoning", tts_reasoning_node)
    graph.add_node("increase_budget", increase_budget_node)
    graph.add_node("try_different_strategy", try_different_strategy_node)
    graph.add_node("accept", lambda s: {**s, "status": "accepted"})
    graph.add_node("escalate", lambda s: {**s, "status": "escalated"})

    # Set entry point
    graph.set_entry_point("tts_reasoning")

    # Add conditional routing
    graph.add_conditional_edges(
        "tts_reasoning",
        route_by_uncertainty,
        {
            "accept": "accept",
            "retry_budget": "increase_budget",
            "retry_strategy": "try_different_strategy",
            "escalate": "escalate"
        }
    )

    # Retry loops back to reasoning
    graph.add_edge("increase_budget", "tts_reasoning")
    graph.add_edge("try_different_strategy", "tts_reasoning")

    # Terminal nodes
    graph.add_edge("accept", END)
    graph.add_edge("escalate", END)

    return graph.compile()

# Usage example
def run_with_uncertainty_routing(question: str) -> AgentState:
    """Run question through uncertainty-aware agent."""
    graph = build_uncertainty_aware_graph()

    initial_state: AgentState = {
        "question": question,
        "answer": None,
        "uncertainty": 1.0,
        "confidence": 0.0,
        "reasoning": None,
        "attempts": 0,
        "budget": 8,
        "strategy": "deepconf",
        "status": "pending"
    }

    final_state = graph.invoke(initial_state)
    return final_state
```

**Pros**:
- Intelligent uncertainty-based routing
- Automatic retry with increased compute
- Strategy fallback mechanism
- Human escalation path
- Visible decision flow

**Cons**:
- More complex to implement
- Requires tuning uncertainty thresholds
- Potentially higher latency (retries)
- Still in-process (same limitations as Approach 1)

**Best for**: Autonomous agents, production systems with reliability requirements, human-in-the-loop workflows

---

### Approach 3: REST API + LangGraph

**Description**: Extend existing `service_app/` to expose TTS with uncertainty via REST API. LangGraph tools call this API.

**Architecture**:
```
┌─────────────────────────────────────────────────────────────┐
│                    LangGraph Agent                          │
│                    (any machine)                            │
│  ┌─────────────────┐                                        │
│  │   TTS API Tool  │────── HTTP ──────┐                     │
│  └─────────────────┘                  │                     │
└───────────────────────────────────────│─────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────┐
│                 TTS Service (service_app/)                  │
│                 (GPU server / container)                    │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │  FastAPI        │───►│  TTS Strategies │                │
│  │  /v1/tts/       │    │  (llm_tts)      │                │
│  └─────────────────┘    └─────────────────┘                │
└─────────────────────────────────────────────────────────────┘
```

**Implementation**:

**1. Extend service_app with uncertainty endpoint:**

```python
# service_app/routers/tts.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Literal

router = APIRouter(prefix="/v1/tts", tags=["tts"])

class TTSRequest(BaseModel):
    prompt: str = Field(description="The question or problem to solve")
    strategy: Literal["deepconf", "self_consistency", "best_of_n"] = "deepconf"
    budget: int = Field(default=8, ge=1, le=64, description="Number of traces")
    model: str = Field(default="openai/gpt-4o-mini", description="Model to use")

class TTSResponse(BaseModel):
    answer: str
    confidence: float = Field(ge=0, le=1)
    uncertainty: float = Field(ge=0, le=1)
    reasoning: str
    num_traces: int
    consensus_ratio: float
    strategy_used: str
    model_used: str
    tokens_used: int

@router.post("/generate", response_model=TTSResponse)
async def generate_with_uncertainty(request: TTSRequest) -> TTSResponse:
    """
    Generate answer using TTS strategy with uncertainty quantification.

    Returns structured response with answer and confidence metrics.
    """
    try:
        # Initialize strategy based on request
        strategy = create_strategy(request.strategy, request.budget)
        model = create_model(request.model)

        # Run generation
        result = await strategy.generate_async(
            prompt=request.prompt,
            model=model
        )

        return TTSResponse(
            answer=result.answer,
            confidence=result.confidence,
            uncertainty=1 - result.confidence,
            reasoning=result.best_trace,
            num_traces=result.num_traces,
            consensus_ratio=result.consensus_ratio,
            strategy_used=request.strategy,
            model_used=request.model,
            tokens_used=result.total_tokens
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "strategies": ["deepconf", "self_consistency", "best_of_n"]}
```

**2. LangGraph tool that calls the API:**

```python
# llm_tts/integrations/langchain_api_tool.py

import httpx
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Literal

TTS_SERVICE_URL = "http://localhost:8000"  # Configure via env

class TTSAPIResult(BaseModel):
    """Result from TTS API service."""
    answer: str
    confidence: float
    uncertainty: float
    reasoning: str
    num_traces: int
    consensus_ratio: float
    strategy_used: str
    model_used: str
    tokens_used: int

@tool
async def tts_reasoning_api(
    question: str,
    strategy: Literal["deepconf", "self_consistency", "best_of_n"] = "deepconf",
    budget: int = 8
) -> TTSAPIResult:
    """
    Solve a reasoning problem using TTS service with uncertainty quantification.

    Calls external TTS service API. Use when you need:
    - High-confidence answers with uncertainty estimates
    - Offloaded computation to GPU server
    - Consistent uncertainty metrics across agents

    Args:
        question: The problem to solve
        strategy: TTS strategy (deepconf, self_consistency, best_of_n)
        budget: Number of reasoning traces (8-32 recommended)

    Returns:
        TTSAPIResult with answer, confidence, uncertainty, and metadata
    """
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{TTS_SERVICE_URL}/v1/tts/generate",
            json={
                "prompt": question,
                "strategy": strategy,
                "budget": budget
            }
        )
        response.raise_for_status()
        data = response.json()

    return TTSAPIResult(**data)

# Synchronous version for non-async contexts
@tool
def tts_reasoning_api_sync(
    question: str,
    strategy: Literal["deepconf", "self_consistency", "best_of_n"] = "deepconf",
    budget: int = 8
) -> TTSAPIResult:
    """Synchronous version of tts_reasoning_api."""
    import httpx

    with httpx.Client(timeout=120.0) as client:
        response = client.post(
            f"{TTS_SERVICE_URL}/v1/tts/generate",
            json={
                "prompt": question,
                "strategy": strategy,
                "budget": budget
            }
        )
        response.raise_for_status()
        data = response.json()

    return TTSAPIResult(**data)
```

**3. Docker deployment:**

```yaml
# docker-compose.langgraph.yml

version: '3.8'

services:
  tts-service:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENROUTER_API_KEY=${OPENROUTER_API_KEY}
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/v1/tts/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  langgraph-agent:
    build: ./agent
    environment:
      - TTS_SERVICE_URL=http://tts-service:8000
    depends_on:
      tts-service:
        condition: service_healthy
```

**Pros**:
- Decoupled architecture (scale independently)
- GPU isolation (TTS on GPU server, agent anywhere)
- Multiple agents can share one TTS service
- Language agnostic (any client can call API)
- Production-ready with health checks
- Easy horizontal scaling

**Cons**:
- Network latency overhead
- More complex deployment
- Need to handle service availability
- Additional infrastructure (Docker, load balancer)

**Best for**: Production deployments, multi-agent systems, microservices architecture, teams with separate ML and application engineers

---

## Comparison Summary

| Aspect | Approach 1: Tool Wrapper | Approach 2: Conditional Routing | Approach 3: REST API |
|--------|--------------------------|--------------------------------|---------------------|
| **Complexity** | Low | Medium | High |
| **Setup Time** | Hours | Days | Days-Week |
| **Latency** | Lowest | Medium (retries) | Higher (network) |
| **Scalability** | Single process | Single process | Horizontal |
| **GPU Sharing** | Shared with agent | Shared with agent | Isolated |
| **Debugging** | Easy | Medium | Harder |
| **Production Ready** | No | Partial | Yes |
| **Best Use Case** | Research/Prototyping | Autonomous agents | Production systems |

---

## Recommended Path

### Phase 1: Research & Prototyping (Approach 1)
- Implement basic LangGraph tool wrapper
- Test with DeepConf strategy (already has confidence)
- Validate uncertainty metrics are meaningful
- **Timeline**: Start here

### Phase 2: Intelligent Routing (Approach 2)
- Add conditional routing based on uncertainty
- Tune thresholds on validation set
- Implement human escalation path
- **Timeline**: After validating Approach 1

### Phase 3: Production Deployment (Approach 3)
- Extend service_app with uncertainty endpoints
- Deploy as microservice
- Add monitoring, logging, rate limiting
- **Timeline**: When moving to production

---

## Implementation Checklist

### Immediate (Approach 1)
- [ ] Create `llm_tts/integrations/` directory
- [ ] Implement `langchain_tool.py` with TTSResult schema
- [ ] Add adapter for DeepConf strategy
- [ ] Write unit tests
- [ ] Create example notebook

### Short-term (Approach 2)
- [ ] Implement `langgraph_nodes.py`
- [ ] Define uncertainty thresholds
- [ ] Add strategy fallback logic
- [ ] Test routing behavior
- [ ] Document threshold tuning

### Medium-term (Approach 3)
- [ ] Extend `service_app/` with `/v1/tts/` endpoints
- [ ] Add async support
- [ ] Implement health checks
- [ ] Create Docker compose for deployment
- [ ] Add monitoring (Prometheus metrics)

---

## Example: Complete LangGraph Agent

```python
# examples/langgraph_agent.py

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from llm_tts.integrations.langchain_tool import tts_reasoning

# Create agent with TTS tool
llm = ChatOpenAI(model="gpt-4o-mini")
tools = [tts_reasoning]
llm_with_tools = llm.bind_tools(tools)

def agent_node(state):
    """Agent decides whether to use TTS or respond directly."""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": messages + [response]}

def should_use_tool(state):
    """Check if agent wants to use TTS tool."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return "end"

# Build graph
graph = StateGraph({"messages": list})
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(tools))

graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_use_tool, {
    "tools": "tools",
    "end": END
})
graph.add_edge("tools", "agent")

agent = graph.compile()

# Run
result = agent.invoke({
    "messages": [{"role": "user", "content": "What is 15% of 240? I need a confident answer."}]
})
print(result)
```

---

## Client Integration: ChatTTS vs Logprobs Tunnel

When integrating with LangChain, there are two ways to pass `tts_metadata` (confidence, uncertainty) to the client:

### Option A: ChatTTS (Custom ChatModel) - RECOMMENDED

```python
from llm_tts.integrations import ChatTTS

llm = ChatTTS(
    base_url="http://localhost:8000/v1",
    model="openai/gpt-4o-mini",
    tts_strategy="self_consistency",
    tts_budget=8,
)

msg = llm.invoke("What is 15% of 240?")

# Clean, semantic access
confidence = msg.response_metadata["tts_metadata"]["confidence"]
uncertainty = msg.response_metadata["tts_metadata"]["uncertainty"]
```

**Pros:**
- Clear semantics: `response_metadata["tts_metadata"]` is obviously TTS-specific
- No dependence on LangChain/OpenAI internals
- Easy to extend (add fields, logging, etc.)
- Future-proof - won't break with LangChain updates
- Self-documenting code

**Cons:**
- Requires ~40 lines of wrapper code
- Users must import `ChatTTS` instead of `ChatOpenAI`
- Need to maintain the wrapper

### Option B: Logprobs Tunnel (Hack)

Server returns `tts_metadata` inside the `logprobs` field:

```json
{
  "choices": [{
    "message": {"content": "..."},
    "logprobs": {
      "content": [...],
      "tts_metadata": {"confidence": 0.85, "uncertainty": 0.15}
    }
  }]
}
```

Client uses standard `ChatOpenAI`:

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy",
    model="openai/gpt-4o-mini",
)

msg = llm.invoke("What is 15% of 240?")

# Hacky access through logprobs
tts_meta = msg.response_metadata["logprobs"]["tts_metadata"]
confidence = tts_meta["confidence"]
```

**Pros:**
- No custom class needed
- Uses standard `ChatOpenAI`
- Works with existing chains/agents
- Quick proof-of-concept

**Cons:**
- **Semantic abuse of logprobs** (meant for token probabilities)
- Potential collisions if you need real logprobs
- **Coupled to LangChain internals** (may break in future versions)
- Confusing for other developers
- Brittle - relies on undocumented behavior

### Comparison Table

| Criterion | ChatTTS (Option A) | Logprobs Tunnel (Option B) |
|-----------|-------------------|---------------------------|
| Setup complexity | ~40 lines | None |
| Uses standard ChatOpenAI | No | Yes |
| Semantic clarity | Excellent | Poor |
| Future-proof | Yes | No |
| Maintainability | High | Low |
| Production ready | Yes | No (hack) |
| Works with existing agents | Yes | Yes |

### Recommendation

- **For production / maintainable code**: Use **ChatTTS** (Option A)
- **For quick PoC / internal tool**: Logprobs tunnel (Option B) is acceptable
- **If you don't need LangChain**: Use direct API calls

---

## References

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Tools](https://python.langchain.com/docs/modules/tools/)
- [DeepConf Paper](docs/deepconf/DeepConf.md)
- [llm-tts-service README](../README.md)
