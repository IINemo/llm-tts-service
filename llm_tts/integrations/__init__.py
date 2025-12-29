# LangChain integration for llm-tts-service
#
# Provides ChatTTS - a custom LangChain ChatModel that calls the TTS service
# and returns uncertainty metrics in response_metadata.

try:
    from llm_tts.integrations.langchain_chat_model import ChatTTS

    LANGCHAIN_AVAILABLE = True
except ImportError:
    ChatTTS = None
    LANGCHAIN_AVAILABLE = False

__all__ = [
    "ChatTTS",
    "LANGCHAIN_AVAILABLE",
]
