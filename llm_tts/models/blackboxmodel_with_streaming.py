import logging
from typing import Dict, List, Optional, Tuple

import openai
from lm_polygraph import BlackboxModel
from lm_polygraph.utils.generation_parameters import GenerationParameters

from llm_tts.early_stopping import ConfidenceEarlyStopping, EarlyStopping

log = logging.getLogger(__name__)


class BlackboxModelWithStreaming(BlackboxModel):
    """
    BlackboxModel subclass that supports multiple streaming generation modes.

    This class extends lm-polygraph's BlackboxModel with streaming capabilities
    and unified early stopping support for different TTS strategies.

    The model can be configured with any EarlyStopping condition:
    - ConfidenceEarlyStopping for DeepConf
    - BoundaryEarlyStopping for Best-of-N
    - CompositeEarlyStopping for hybrid approaches
    - Custom implementations of EarlyStopping

    Args:
        openai_api_key (str): OpenAI API key.
        model_path (str): Model name or path.
        hf_api_token (str): HuggingFace API token (unused here).
        generation_parameters (GenerationParameters): Generation parameters.
        supports_logprobs (bool): Whether logprobs are supported.
        base_url (str): Custom API base URL (e.g., for OpenRouter).
        early_stopping (EarlyStopping): Optional early stopping condition.
    """

    def __init__(
        self,
        openai_api_key: str = None,
        model_path: str = None,
        hf_api_token: str = None,
        generation_parameters: GenerationParameters = GenerationParameters(),
        supports_logprobs: bool = False,
        base_url: Optional[str] = None,
        early_stopping: Optional[EarlyStopping] = None,
    ):
        super().__init__(
            openai_api_key=openai_api_key,
            model_path=model_path,
            hf_api_token=hf_api_token,
            generation_parameters=generation_parameters,
            supports_logprobs=supports_logprobs,
        )
        # Create client with optional custom base_url (e.g., OpenRouter)
        client_kwargs = {"api_key": openai_api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = openai.OpenAI(**client_kwargs)

        # Override parent's openai_api for non-streaming calls
        self.openai_api = self.client

        # Store early stopping configuration
        self.early_stopping = early_stopping

    def generate_texts(self, chats: List[List[Dict[str, str]]], **args) -> List[dict]:
        """
        Generate texts, optionally returning token logprobs.

        Args:
            chats: List of chat message lists
            **args: Generation parameters including:
                - output_scores (bool): Request logprobs
                - max_new_tokens (int): Max tokens to generate
                - temperature (float): Sampling temperature
                - top_logprobs (int): Number of top logprobs per token

        Returns:
            List[dict] with keys: 'text' and optionally 'logprobs'
        """
        max_new_tokens = args.get("max_new_tokens", 512)
        temperature = args.get("temperature", 0.7)

        early_stopping = args.get("early_stopping", self.early_stopping)
        needs_logprobs = isinstance(
            early_stopping, ConfidenceEarlyStopping
        ) or args.get("output_scores", False)
        top_logprobs = args.get("top_logprobs", 20)

        results: List[dict] = []
        for chat in chats:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_path,
                    messages=chat,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    logprobs=needs_logprobs,
                    top_logprobs=top_logprobs if needs_logprobs else None,
                )

                text = response.choices[0].message.content

                entry: Dict = {"text": text}

                if needs_logprobs:
                    token_confidence_data: List[Dict] = []
                    if (
                        hasattr(response.choices[0], "logprobs")
                        and response.choices[0].logprobs
                    ):
                        logprobs_obj = response.choices[0].logprobs
                        if hasattr(logprobs_obj, "content") and logprobs_obj.content:
                            for token_info in logprobs_obj.content:
                                token_confidence_data.append(
                                    {
                                        "token": token_info.token,
                                        "logprob": token_info.logprob,
                                        "top_logprobs": [
                                            {"token": t.token, "logprob": t.logprob}
                                            for t in token_info.top_logprobs
                                        ],
                                    }
                                )
                    entry["logprobs"] = token_confidence_data

                results.append(entry)
            except Exception as e:
                log.error(f"Streaming generation failed: {e}")
                results.append({"text": "", "error": str(e)})

        return results

    def generate_with_confidence(
        self, prompt: str, max_tokens: int = 512, temperature: float = 0.7, **kwargs
    ) -> Tuple[str, Optional[List[Dict]]]:
        """
        Generate text and extract token-level confidence scores.

        Args:
            prompt: Input prompt (string or message list)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional provider-specific parameters

        Returns:
            Tuple of (generated_text, token_confidence_data)
            where token_confidence_data contains logprobs for each token
        """
        try:
            # Use openai_api (parent's client) if available, otherwise use self.client
            client = getattr(self, "openai_api", self.client)

            # Handle both string prompts and message lists
            if isinstance(prompt, list):
                messages = prompt
            else:
                messages = [{"role": "user", "content": prompt}]

            response = client.chat.completions.create(
                model=self.model_path,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                logprobs=True,
                top_logprobs=20,  # Max allowed by OpenAI
                **kwargs,
            )

            # Extract generated text
            generated_text = response.choices[0].message.content

            # Extract token confidence data
            token_confidence_data = []
            if (
                hasattr(response.choices[0], "logprobs")
                and response.choices[0].logprobs
            ):
                logprobs_obj = response.choices[0].logprobs

                # Check if it has 'content' attribute
                if hasattr(logprobs_obj, "content") and logprobs_obj.content:
                    for token_info in logprobs_obj.content:
                        token_data = {
                            "token": token_info.token,
                            "logprob": token_info.logprob,
                            "top_logprobs": [
                                {"token": t.token, "logprob": t.logprob}
                                for t in token_info.top_logprobs
                            ],
                        }
                        token_confidence_data.append(token_data)
                else:
                    log.warning(
                        "Logprobs object exists but has no 'content' attribute or it's empty"
                    )
            else:
                log.warning(
                    f"No logprobs in response! Model '{self.model_path}' "
                    "may not support logprobs"
                )

            return generated_text, token_confidence_data

        except Exception as e:
            log.error(f"Generation with confidence failed: {e}")
            raise

    def tokenize(self, texts: List[str]) -> List[List[str]]:
        return [e.split() for e in texts]
