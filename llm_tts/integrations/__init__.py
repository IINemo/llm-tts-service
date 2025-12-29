# LangChain integration for llm-tts-service
#
# Provides ChatTTS - a custom LangChain ChatModel that calls the TTS service
# and returns uncertainty metrics in response_metadata.

try:
    from llm_tts.integrations.langchain_chat_model import ChatTTS, create_chat_tts

    LANGCHAIN_AVAILABLE = True
except ImportError:
    ChatTTS = None
    create_chat_tts = None
    LANGCHAIN_AVAILABLE = False

__all__ = [
    "ChatTTS",
    "create_chat_tts",
    "LANGCHAIN_AVAILABLE",
]
