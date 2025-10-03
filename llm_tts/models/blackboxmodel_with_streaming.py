from typing import Dict, List, Optional

import openai
from lm_polygraph import BlackboxModel
from lm_polygraph.utils.generation_parameters import GenerationParameters

from llm_tts.step_boundary_detector import StepBoundaryDetector


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
        # Create client with optional custom base_url (e.g., OpenRouter)
        client_kwargs = {"api_key": openai_api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = openai.OpenAI(**client_kwargs)

        # Override parent's openai_api for non-streaming calls
        self.openai_api = self.client

        self.boundary_detector = boundary_detector
        self.stop_args = {}

    def generate_texts(self, chats: List[List[Dict[str, str]]], **args) -> List[dict]:
        """
        Streams completions for each input text, returning step/trajectory info per input.

        If output_scores=True is in args, delegates to parent's non-streaming method
        for logprobs support (used by DeepConf).

        Args:
            chats (List[List[Dict[str, str]]]): List of chat message lists.
            **args: Additional arguments.

        Returns:
            List[dict] with step info (streaming) or List[str] (non-streaming with logprobs)
        """
        # Check if logprobs are requested (DeepConf offline mode)
        if args.get("output_scores", False):
            # Use parent's non-streaming method with logprobs support
            return super().generate_texts(chats, **args)

        # Otherwise use streaming mode
        results = []
        for chat in chats:
            buffer = []
            with self.client.responses.stream(
                model=self.model_path,
                input=chat,
                **self.stop_args,
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
                                    "trajectory_complete": (
                                        self.boundary_detector.is_trajectory_complete(
                                            text_now
                                        )
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

    def tokenize(self, texts: List[str]) -> List[List[str]]:
        return [e.split() for e in texts]
