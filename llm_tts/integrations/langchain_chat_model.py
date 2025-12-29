"""
Custom LangChain ChatModel for llm-tts-service.

This is the PROPER way to integrate with LangChain - a custom BaseChatModel
that returns tts_metadata in response_metadata directly.

Usage:
    from llm_tts.integrations import ChatTTS

    llm = ChatTTS(
        base_url="http://localhost:8000/v1",
        model="openai/gpt-4o-mini",
        tts_strategy="self_consistency",
        tts_budget=8,
    )

    msg = llm.invoke("What is 15% of 240?")
    print(msg.content)
    print(msg.response_metadata["tts_metadata"]["confidence"])
"""

import logging
from typing import Any, Dict, Iterator, List, Optional

import httpx
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

log = logging.getLogger(__name__)


class ChatTTS(BaseChatModel):
    """
    LangChain ChatModel for llm-tts-service with proper tts_metadata support.

    This custom model calls your llm-tts-service and returns tts_metadata
    (confidence, uncertainty, strategy) in response_metadata where it belongs.

    Attributes:
        base_url: URL of your llm-tts-service (e.g., "http://localhost:8000/v1")
        api_key: API key (optional, your service may not require it)
        model: Model name (e.g., "openai/gpt-4o-mini")
        tts_strategy: TTS strategy to use ("self_consistency", "deepconf", etc.)
        tts_budget: Number of reasoning traces to generate
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
    """

    base_url: str = Field(default="http://localhost:8000/v1")
    api_key: str = Field(default="dummy")
    model: str = Field(default="openai/gpt-4o-mini")
    tts_strategy: str = Field(default="self_consistency")
    tts_budget: int = Field(default=8)
    tts_mode: Optional[str] = Field(default=None)
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=512)
    timeout: float = Field(default=120.0)

    @property
    def _llm_type(self) -> str:
        return "chat-tts"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "base_url": self.base_url,
            "model": self.model,
            "tts_strategy": self.tts_strategy,
            "tts_budget": self.tts_budget,
        }

    def _convert_messages(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        """Convert LangChain messages to OpenAI format."""
        result = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                result.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                result.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                result.append({"role": "assistant", "content": msg.content})
            else:
                result.append({"role": "user", "content": str(msg.content)})
        return result

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a response using llm-tts-service."""

        # Build request
        request_body = {
            "model": self.model,
            "messages": self._convert_messages(messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "tts_strategy": self.tts_strategy,
            "num_paths": self.tts_budget,  # Service expects num_paths
        }

        if self.tts_mode:
            request_body["tts_mode"] = self.tts_mode

        if stop:
            request_body["stop"] = stop

        # Call service
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.base_url}/chat/completions",
                json=request_body,
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            response.raise_for_status()
            data = response.json()

        # Extract response
        choice = data["choices"][0]
        content = choice["message"]["content"]
        finish_reason = choice.get("finish_reason", "stop")

        # Extract tts_metadata - THIS IS THE KEY DIFFERENCE
        # We put it in response_metadata where it semantically belongs
        tts_metadata = choice.get("tts_metadata", {})

        # Build response_metadata with all relevant info
        response_metadata = {
            "tts_metadata": tts_metadata,
            "model": data.get("model", self.model),
            "finish_reason": finish_reason,
        }

        # Add usage if available
        if "usage" in data:
            response_metadata["usage"] = data["usage"]

        # Create AIMessage with proper response_metadata
        message = AIMessage(
            content=content,
            response_metadata=response_metadata,
        )

        return ChatResult(
            generations=[ChatGeneration(message=message)],
            llm_output={"model": self.model},
        )

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async generate - uses sync for now."""
        # TODO: Implement proper async with httpx.AsyncClient
        return self._generate(messages, stop, run_manager, **kwargs)


# Convenience function
def create_chat_tts(
    base_url: str = "http://localhost:8000/v1",
    model: str = "openai/gpt-4o-mini",
    tts_strategy: str = "self_consistency",
    tts_budget: int = 8,
    **kwargs,
) -> ChatTTS:
    """
    Create a ChatTTS instance with sensible defaults.

    Example:
        from llm_tts.integrations import create_chat_tts

        llm = create_chat_tts(tts_strategy="deepconf", tts_budget=16)
        msg = llm.invoke("Solve this step by step: What is 25% of 80?")
        print(f"Answer: {msg.content}")
        print(f"Confidence: {msg.response_metadata['tts_metadata']['confidence']}")
    """
    return ChatTTS(
        base_url=base_url,
        model=model,
        tts_strategy=tts_strategy,
        tts_budget=tts_budget,
        **kwargs,
    )
