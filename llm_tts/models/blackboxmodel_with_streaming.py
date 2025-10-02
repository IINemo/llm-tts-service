from typing import Dict, List, Optional

import openai
from lm_polygraph import BlackboxModel
from lm_polygraph.utils.generation_parameters import GenerationParameters

from llm_tts.step_detection import StepBoundaryDetector


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
    ):
        super().__init__(
            openai_api_key=openai_api_key,
            model_path=model_path,
            hf_api_token=hf_api_token,
            generation_parameters=generation_parameters,
            supports_logprobs=supports_logprobs,
        )
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
            token_count = 0
            print("====================")
            print(chat)
            print("**********************")
            with self.client.responses.stream(
                model=self.model_path,
                input=chat,
                **self.stop_args,
            ) as stream:
                for event in stream:
                    if event.type == "response.output_text.delta":
                        delta = event.delta
                        buffer.append(delta)
                        # rough proxy; use tiktoken for precise token counting
                        token_count += 1

                        current_text = "".join(buffer)

                        if (
                            self.boundary_detector
                            and self.boundary_detector.is_step_complete(
                                current_text, token_count
                            )
                        ):
                            stream.close()
                            step_text = self.boundary_detector.extract_step_text(
                                current_text
                            )
                            trajectory_done = (
                                self.boundary_detector.is_trajectory_complete(
                                    current_text
                                )
                            )
                            results.append(
                                {
                                    "step_text": step_text,
                                    "raw_collected": current_text,
                                    "token_count_guess": token_count,
                                    "trajectory_complete": trajectory_done,
                                    "reason": "boundary-detected",
                                }
                            )
                            break  # Stop streaming for this input

                    elif event.type in ("response.completed", "response.error"):
                        # Natural end or error; fall through and return what we have
                        break

                    else:
                        # If we exited the loop without boundary, return whatever we collected
                        current_text = "".join(buffer)
                        if self.boundary_detector:
                            results.append(
                                {
                                    "step_text": (
                                        self.boundary_detector.extract_step_text(
                                            current_text
                                        )
                                    ),
                                    "raw_collected": current_text,
                                    "token_count_guess": token_count,
                                    "trajectory_complete": (
                                        self.boundary_detector.is_trajectory_complete(
                                            current_text
                                        )
                                    ),
                                    "reason": "stream-ended",
                                }
                            )
                        else:
                            results.append(
                                {
                                    "step_text": current_text,
                                    "raw_collected": current_text,
                                    "token_count_guess": token_count,
                                    "trajectory_complete": False,
                                    "reason": "stream-ended",
                                }
                            )
        return results

    def tokenize(self, texts: List[str]) -> List[List[str]]:
        return [e.split() for e in texts]
