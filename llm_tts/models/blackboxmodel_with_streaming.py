import logging
from typing import Dict, List, Optional, Tuple

import openai
from lm_polygraph import BlackboxModel
from lm_polygraph.utils.generation_parameters import GenerationParameters

from llm_tts.step_boundary_detector import StepBoundaryDetector

log = logging.getLogger(__name__)


class BlackboxModelWithStreaming(BlackboxModel):
    """
    BlackboxModel subclass that supports streaming generation using OpenAI's API,
    with step boundary detection for early stopping.

    Args:
        openai_api_key (str): OpenAI API key.
        model_path (str): Model name or path.
        hf_api_token (str): HuggingFace API token (unused here).
        generation_parameters (GenerationParameters): Generation parameters.
        supports_logprobs (bool): Whether logprobs are supported.
        boundary_detector (StepBoundaryDetector): Detector for step/trajectory boundaries.
    """

    def __init__(
        self,
        openai_api_key: str = None,
        model_path: str = None,
        hf_api_token: str = None,
        generation_parameters: GenerationParameters = GenerationParameters(),
        supports_logprobs: bool = False,
        boundary_detector: Optional[StepBoundaryDetector] = None,
        base_url: Optional[str] = None,
    ):
        super().__init__(
            openai_api_key=openai_api_key,
            model_path=model_path,
            hf_api_token=hf_api_token,
            generation_parameters=generation_parameters,
            supports_logprobs=supports_logprobs,
        )
        # Create OpenAI client with optional base_url for OpenRouter
        if base_url:
            self.client = openai.OpenAI(api_key=openai_api_key, base_url=base_url)
            # Also update the parent's openai_api client for logprobs support
            self.openai_api = openai.OpenAI(api_key=openai_api_key, base_url=base_url)
        else:
            self.client = openai.OpenAI(api_key=openai_api_key)
        self.boundary_detector = boundary_detector
        self.stop_args = {}

    def generate_texts(self, chats: List[List[Dict[str, str]]], **args) -> List[dict]:
        """
        Streams completions for each input text, returning step/trajectory info per input.

        Args:
            input_texts (List[str]): List of user prompts.
            **args: Additional arguments (currently unused).

        Returns:
            List[dict]: List of dicts with step/trajectory info for each input.
        """

        results = []
        for chat in chats:
            buffer = []
            with self.client.responses.stream(
                model=self.model_path,
                input=chat,
                **self.stop_args,  # TODO: fix / add stop args
            ) as stream:
                buffer = []
                boundary_fired = False

                for event in stream:
                    t = getattr(event, "type", None)

                    if t == "response.output_text.delta":
                        delta = event.delta  # only exists for this type
                        buffer.append(delta)

                        if (
                            self.boundary_detector
                            and self.boundary_detector.is_step_complete("".join(buffer))
                        ):
                            stream.close()  # early stop
                            text_now = "".join(buffer)
                            results.append(
                                {
                                    "step_text": self.boundary_detector.extract_step_text(
                                        text_now
                                    ),
                                    "raw_collected": text_now,
                                    "trajectory_complete": self.boundary_detector.is_trajectory_complete(
                                        text_now
                                    ),
                                    "reason": "boundary-detected",
                                }
                            )
                            boundary_fired = True
                            break

                    elif t in ("response.completed", "response.error"):
                        # finalize after the loop
                        break

                    else:
                        # ignore unrelated event types; keep streaming
                        continue

                if not boundary_fired:
                    text_now = "".join(buffer)
                    results.append(
                        {
                            "step_text": (
                                self.boundary_detector.extract_step_text(text_now)
                                if self.boundary_detector
                                else text_now
                            ),
                            "raw_collected": text_now,
                            "trajectory_complete": (
                                self.boundary_detector.is_trajectory_complete(text_now)
                                if self.boundary_detector
                                else False
                            ),
                            "reason": (
                                "stream-ended"
                                if t == "response.completed"
                                else (
                                    "stream-error"
                                    if t == "response.error"
                                    else "ended-unknown"
                                )
                            ),
                        }
                    )

        return results

    def generate_with_confidence(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float = 0.7,
        num_return_sequences: int = 1,
        **kwargs,
    ) -> Tuple[str, Optional[List[Dict]]]:
        """
        Generate text and extract token-level confidence scores.

        Args:
            prompt: Input prompt (string or message list)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional provider-specific parameters

        Returns:
            List of (generated_text, token_confidence_data) pairs
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
                # top_logprobs=20,  # not every model / provider supports excatcly 20
                # so it is better to specify it and pass in **kwargs
                **kwargs,
            )

            results = []
            for choice in response.choices:
                # Extract generated text for this choice
                generated_text = choice.message.content

                # Extract token confidence data for this choice
                token_confidence_data: List[Dict] = []
                if hasattr(choice, "logprobs") and choice.logprobs:
                    logprobs_obj = choice.logprobs
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
                        f"No logprobs in response choice! Model '{self.model_path}' may not support logprobs"
                    )

                results.append((generated_text, token_confidence_data))

            return results

        except Exception as e:
            log.error(f"Generation with confidence failed: {e}")
            raise

    def tokenize(self, texts: List[str]) -> List[List[str]]:
        return [e.split() for e in texts]
