import logging
from typing import Dict, List, Optional

import openai
from lm_polygraph import BlackboxModel
from lm_polygraph.utils.generation_parameters import GenerationParameters

from llm_tts.early_stopping import (
    BoundaryEarlyStopping,
    ConfidenceEarlyStopping,
    EarlyStopping,
)

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
        Generate texts using streaming.

        Args:
            chats: List of chat message lists
            **args: Generation parameters including:
                - output_scores (bool): Request logprobs
                - max_new_tokens (int): Max tokens to generate
                - temperature (float): Sampling temperature

        Returns:
            List of generation results with streaming
        """
        # Extract parameters
        max_new_tokens = args.get("max_new_tokens", 512)
        temperature = args.get("temperature", 0.7)

        # Use model's early_stopping (can be overridden by args)
        early_stopping = args.get("early_stopping", self.early_stopping)

        # Determine if we need logprobs
        needs_logprobs = isinstance(
            early_stopping, ConfidenceEarlyStopping
        ) or args.get(  # Confidence needs logprobs
            "output_scores", False
        )  # Explicit request

        results = []
        for chat in chats:
            # Create streaming request
            response = self.client.chat.completions.create(
                model=self.model_path,
                messages=chat,
                max_tokens=max_new_tokens,
                temperature=temperature,
                stream=True,
                logprobs=needs_logprobs,
                top_logprobs=20 if needs_logprobs else None, # TODO:
            )

            accumulated_text = ""
            all_logprobs = []
            token_count = 0
            stopped_early = False
            stop_reason = None

            for chunk in response:
                if not chunk.choices:
                    continue

                choice = chunk.choices[0]

                # Extract token
                current_token = None
                if hasattr(choice, "delta") and choice.delta:
                    delta = choice.delta
                    current_token = getattr(delta, "content", None)

                    if current_token:
                        accumulated_text += current_token
                        token_count += 1

                # Extract logprobs if available
                current_logprob = None
                current_top_logprobs = []
                if needs_logprobs and hasattr(choice, "logprobs") and choice.logprobs:
                    logprobs_obj = choice.logprobs
                    if hasattr(logprobs_obj, "content") and logprobs_obj.content:
                        for token_info in logprobs_obj.content:
                            logprob_data = {
                                "token": token_info.token,
                                "logprob": token_info.logprob,
                                "top_logprobs": [
                                    {"token": t.token, "logprob": t.logprob}
                                    for t in token_info.top_logprobs
                                ],
                            }
                            all_logprobs.append(logprob_data)
                            current_logprob = token_info.logprob
                            current_top_logprobs = logprob_data["top_logprobs"]

                # Check early stopping condition
                if early_stopping is not None:
                    state = {
                        "text": accumulated_text,
                        "token_count": token_count,
                        "logprob": current_logprob,
                        "top_logprobs": current_top_logprobs,
                    }

                    if early_stopping.should_stop(state):
                        stopped_early = True
                        stop_reason = early_stopping.get_reason()
                        break

                # Check if generation finished
                if hasattr(choice, "finish_reason") and choice.finish_reason:
                    break

            # Build result
            result = {"text": accumulated_text}

            # Add logprobs if collected
            if needs_logprobs:
                result["logprobs"] = all_logprobs

            # Add stopping info for DeepConf
            if isinstance(early_stopping, ConfidenceEarlyStopping):
                result["stopped_early"] = stopped_early

            # Add boundary detection info for Best-of-N
            if isinstance(early_stopping, BoundaryEarlyStopping):
                detector = early_stopping.detector
                result["step_text"] = detector.extract_step_text(accumulated_text)
                result["raw_collected"] = accumulated_text
                result["trajectory_complete"] = detector.is_trajectory_complete(
                    accumulated_text
                )
                result["reason"] = stop_reason or "stream-ended"

            results.append(result)

        return results

    def tokenize(self, texts: List[str]) -> List[List[str]]:
        return [e.split() for e in texts]
