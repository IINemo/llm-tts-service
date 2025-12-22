# LangChain Integration with llm-tts-service

How to integrate `service_app` with LangChain to get uncertainty metrics alongside answers.

## API Response

The service returns `tts_metadata` with confidence info:

```json
{
  "choices": [{
    "message": {"content": "36"},
    "tts_metadata": {
      "strategy": "deepconf",
      "confidence": 0.85,
      "extracted_answer": "36"
    }
  }]
}
```

## Integration Approaches

### A: Custom ChatModel (Recommended)

```python
from llm_tts.integrations import ChatTTS

llm = ChatTTS(
    base_url="http://localhost:8000/v1",
    model="openai/gpt-4o-mini",
    tts_strategy="deepconf",
    tts_budget=8,
)

msg = llm.invoke("What is 15% of 240?")
confidence = msg.response_metadata["tts_metadata"]["confidence"]
```

**Pros:**
- Clean semantics: `response_metadata["tts_metadata"]`
- Works with LangChain chains and agents
- Easy to extend and maintain
- Future-proof

**Cons:**
- Requires importing custom class (~40 lines)

---

### B: Direct API

```python
import httpx

resp = httpx.post("http://localhost:8000/v1/chat/completions", json={
    "model": "openai/gpt-4o-mini",
    "messages": [{"role": "user", "content": "What is 15% of 240?"}],
    "tts_strategy": "deepconf",
    "tts_budget": 8,
})
data = resp.json()
confidence = data["choices"][0]["tts_metadata"]["confidence"]
```

**Pros:**
- No LangChain dependency
- Full control over API
- Simplest implementation

**Cons:**
- Can't use LangChain agents/chains
- Manual retry/streaming handling

---

### C: Response-level Metadata

The server returns `metadata` at response level (not inside `choices`):

```json
{
  "id": "...",
  "metadata": {"confidence": 0.85, "strategy": "deepconf"},
  "choices": [...]
}
```

This field IS passed through by the OpenAI SDK, but **NOT by LangChain's ChatOpenAI** for standard chat completions. It only works with the new Responses API.

```python
# Works with raw OpenAI SDK:
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
resp = client.chat.completions.create(model="openai/gpt-4o-mini", messages=[...])
print(resp.metadata)  # Works!

# Does NOT work with LangChain ChatOpenAI:
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(base_url="http://localhost:8000/v1", ...)
msg = llm.invoke("...")
print(msg.response_metadata.get("metadata"))  # None - not passed through
```

**Use this approach** if you're using the OpenAI SDK directly, not LangChain.

---

### D: Logprobs Tunnel (Hack)

Tunnel metadata through `logprobs` field - the only choice-level field LangChain passes through.

Server returns:
```json
{"choices": [{"logprobs": {"content": [], "tts_metadata": {"confidence": 0.85}}}]}
```

Client:
```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy",
    model="openai/gpt-4o-mini",
    logprobs=True,
)

msg = llm.invoke("What is 15% of 240?")
confidence = msg.response_metadata["logprobs"]["tts_metadata"]["confidence"]
```

**Pros:**
- Uses standard `ChatOpenAI` - no custom class needed
- Works with existing LangChain chains/agents
- Third-party users just change the URL

**Cons:**
- Semantic abuse of logprobs field (meant for token probabilities)
- Requires server-side modification to embed metadata in logprobs
- May conflict if you need real logprobs
- Coupled to LangChain internals (may break in future versions)

---

### E: Standard ChatOpenAI (Limited)

```python
llm = ChatOpenAI(base_url="http://localhost:8000/v1", ...)
# tts_metadata NOT accessible - LangChain doesn't pass custom fields
```

Not recommended - use ChatTTS (A) or Logprobs Tunnel (D) for metadata access.

---

## Why LangChain Filters Custom Fields

LangChain explicitly whitelists which fields go into `response_metadata`:

**From response level:**
- `token_usage`, `model_name`, `model_provider`, `system_fingerprint`, `id`, `service_tier`

**From choice level:**
- `finish_reason`
- `logprobs` (only standard field passed through)

```python
# LangChain code - only logprobs extracted from choice
if "logprobs" in res:
    generation_info["logprobs"] = res["logprobs"]
```

Custom fields like `tts_metadata` (choice level) or `metadata` (response level) are ignored. The `metadata` whitelist only applies to the new Responses API (`output_version="responses/v1"`).

## Why ChatTTS Bypasses Filtering

`ChatTTS` extends `BaseChatModel` and implements `_generate()` directly - it never calls LangChain's filtering code:

**ChatOpenAI flow (filtered):**
```
llm.invoke() → OpenAI SDK → LangChain._create_chat_result() → WHITELIST FILTER → AIMessage
```

**ChatTTS flow (not filtered):**
```
llm.invoke() → our _generate() → raw httpx.post() → we parse JSON ourselves → we build response_metadata → AIMessage
```

Key code in ChatTTS:
```python
def _generate(self, messages, ...):
    # Raw HTTP request - no SDK
    response = httpx.post(f"{self.base_url}/chat/completions", json=...)
    data = response.json()

    # We extract tts_metadata ourselves
    tts_metadata = data["choices"][0].get("tts_metadata", {})

    # We build response_metadata ourselves - no whitelist
    response_metadata = {"tts_metadata": tts_metadata, ...}

    return ChatResult(generations=[ChatGeneration(
        message=AIMessage(content=..., response_metadata=response_metadata)
    )])
```

LangChain just calls our `_generate()` and trusts whatever `ChatResult` we return.

---

## Comparison

| Approach | Access metadata | Works with agents | LangChain | Production ready |
|----------|----------------|-------------------|-----------|------------------|
| A: ChatTTS | Yes | Yes | Yes (custom) | Yes |
| B: Direct API | Yes | No | No | Yes |
| C: Response metadata | Yes | No | No (SDK only) | Yes |
| D: Logprobs Tunnel | Yes | Yes | Yes (hack) | No |
| E: Standard ChatOpenAI | No | Yes | Yes | No |

## LangChain Agent with Tools

```python
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from llm_tts.integrations import ChatTTS

@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

llm = ChatTTS(
    base_url="http://localhost:8000/v1",
    model="openai/gpt-4o-mini",
    tts_strategy="deepconf",
    tts_budget=8,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, [calculator, search], prompt)
executor = AgentExecutor(agent=agent, tools=[calculator, search])

result = executor.invoke({"input": "What is 15% of 240?"})
print(result["output"])
```

---

## Adaptive TTS Based on Confidence

A tool that switches TTS strategy based on returned confidence:

```python
from langchain_core.tools import tool
from llm_tts.integrations import ChatTTS

@tool
def reason_adaptive(question: str) -> str:
    """
    Answer with adaptive TTS - starts fast, retries with heavier strategy if uncertain.
    """
    # First try: fast strategy
    llm = ChatTTS(tts_strategy="deepconf", tts_budget=4)
    msg = llm.invoke(question)
    confidence = msg.response_metadata["tts_metadata"].get("confidence") or 0

    if confidence >= 0.8:
        return f"Answer: {msg.content} (confidence: {confidence})"

    # Low confidence: retry with heavier strategy
    llm = ChatTTS(tts_strategy="deepconf", tts_budget=16)
    msg = llm.invoke(question)
    confidence = msg.response_metadata["tts_metadata"].get("confidence") or 0

    return f"Answer: {msg.content} (confidence: {confidence}, used heavy strategy)"
```

---

## Confidence-Based Strategy Selection

Chain that selects TTS strategy based on confidence from previous call:

```python
from langchain_core.runnables import RunnableLambda
from llm_tts.integrations import ChatTTS

def adaptive_tts_chain(question: str) -> dict:
    """
    Adaptive chain:
    1. Fast reasoning first
    2. If low confidence -> heavy reasoning
    3. If medium confidence -> verify
    """
    # Step 1: Fast reasoning
    fast_llm = ChatTTS(tts_strategy="deepconf", tts_budget=4)
    msg = fast_llm.invoke(question)
    confidence = msg.response_metadata["tts_metadata"].get("confidence") or 0

    result = {
        "answer": msg.content,
        "confidence": confidence,
        "strategy": "fast",
    }

    # Step 2: Branch based on confidence
    if confidence >= 0.8:
        return result  # Done - high confidence

    if confidence < 0.5:
        # Low confidence - use heavy strategy
        heavy_llm = ChatTTS(tts_strategy="tot", tts_budget=8)
        msg = heavy_llm.invoke(question)
        return {
            "answer": msg.content,
            "confidence": msg.response_metadata["tts_metadata"].get("confidence") or 0,
            "strategy": "heavy",
        }

    # Medium confidence - verify answer
    verify_llm = ChatTTS(tts_strategy="deepconf", tts_budget=16)
    msg = verify_llm.invoke(f"Verify: {result['answer']}")
    return {
        "answer": msg.content,
        "confidence": msg.response_metadata["tts_metadata"].get("confidence") or 0,
        "strategy": "verify",
    }

# Use as runnable
chain = RunnableLambda(adaptive_tts_chain)
result = chain.invoke("What is the integral of x^2 from 0 to 1?")
print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']}")
print(f"Strategy used: {result['strategy']}")
```
