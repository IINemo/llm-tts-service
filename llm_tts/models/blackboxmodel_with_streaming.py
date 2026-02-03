import logging
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
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

        # Store client parameters for recreation
        self._api_key = openai_api_key
        self._base_url = base_url

        # Create persistent client with optional custom base_url (e.g., OpenRouter)
        # Configure timeouts for connection, read, write, and pool
        # Read timeout handles both streaming chunks and non-streaming responses
        from httpx import Timeout

        client_kwargs = {
            "api_key": openai_api_key,
            "timeout": Timeout(
                connect=10.0,  # 10s to establish connection
                read=300.0,  # 60s to receive response/next chunk
                write=10.0,  # 10s to send request
                pool=10.0,  # 10s to get connection from pool
            ),
            "max_retries": 0,  # Disable built-in retries (handled at strategy level)
        }
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = openai.OpenAI(**client_kwargs)

        # Override parent's openai_api for non-streaming calls
        self.openai_api = self.client

        # Create thread pool executor for timeout enforcement
        # Use max_workers=10 to handle concurrent requests in strategies
        self._executor = ThreadPoolExecutor(
            max_workers=10, thread_name_prefix="llm_api"
        )

        # Store early stopping configuration
        self.early_stopping = early_stopping

    def recreate_client(self):
        """
        Recreate the OpenAI client to clear any stuck connections.

        This is useful when API calls timeout, as the connection pool may be in a bad state.
        Creates a fresh client with the same configuration.

        Note: We don't explicitly close the old client because:
        1. If connections are stuck, close() will hang
        2. Python's garbage collector will clean up the old client
        3. The old client's connections will eventually timeout on their own
        """
        from httpx import Timeout

        log.info("[CLIENT] Recreating OpenAI client (abandoning old stuck connections)")

        # Don't close old client - it may hang if connections are stuck
        # Just replace it and let garbage collector handle cleanup
        old_client = self.client

        # Create new client with same parameters
        client_kwargs = {
            "api_key": self._api_key,
            "timeout": Timeout(connect=10.0, read=300.0, write=10.0, pool=10.0),
            "max_retries": 0,
        }
        if self._base_url:
            client_kwargs["base_url"] = self._base_url

        self.client = openai.OpenAI(**client_kwargs)
        self.openai_api = self.client

        log.info(
            "[CLIENT] New client created successfully (old client will be garbage collected)"
        )

        # Mark old client as unreferenced (will be garbage collected)
        del old_client

    def shutdown(self):
        """
        Shutdown the model's resources (executor, client).

        Call this at the end of your program to ensure clean exit.
        """
        log.info("[MODEL] Shutting down model resources...")

        # Shutdown executor
        if hasattr(self, "_executor") and self._executor is not None:
            self._executor.shutdown(wait=False)
            log.info("[MODEL] Executor shut down")

        # Don't explicitly close client as it may hang
        # Let garbage collector handle it

        log.info("[MODEL] Shutdown complete")

    def generate_texts(self, chats: List[List[Dict[str, str]]], **args) -> List[dict]:
        """
        Generate texts using persistent client with timeout protection.

        Args:
            chats: List of chat message lists
            **args: Generation parameters including:
                - timeout: Total timeout in seconds (default: 60)
                - Other params passed to _generate_texts_impl

        Uses ThreadPoolExecutor to enforce total timeout on API calls.
        The persistent client is used for all requests (no new clients created).
        Retry logic should be implemented at the strategy level.
        """
        n = args.get("n", 1)
        timeout = args.pop("timeout", 60)  # Extract timeout, default 60s

        log.info(
            f"[CALL START] generate_texts with n={n} for {len(chats)} chat(s), timeout={timeout}s"
        )

        # Submit to executor with timeout enforcement
        future = self._executor.submit(self._generate_texts_impl, chats, **args)

        try:
            result = future.result(timeout=timeout)
            log.info(f"[CALL SUCCESS] Returning {len(result)} results")
            return result
        except FuturesTimeoutError:
            log.error(f"[TIMEOUT] API call exceeded {timeout}s timeout")
            future.cancel()  # Attempt to cancel (may not work if already running)
            raise openai.APITimeoutError(f"API call timed out after {timeout}s")
        except Exception as e:
            log.error(f"[CALL ERROR] Exception: {type(e).__name__}: {e}")
            raise

    def _generate_texts_impl(
        self, chats: List[List[Dict[str, str]]], **args
    ) -> List[dict]:
        """Internal implementation of generate_texts without timeout wrapper."""
        # Extract parameters
        max_new_tokens = args.get("max_new_tokens", 512)
        temperature = args.get("temperature", 0.7)
        n = args.get("n", 1)

        # Use model's early_stopping (can be overridden by args)
        early_stopping = args.get("early_stopping", self.early_stopping)

        # Determine if we need logprobs
        needs_logprobs = isinstance(
            early_stopping, ConfidenceEarlyStopping
        ) or args.get(  # Confidence needs logprobs
            "output_scores", False
        )  # Explicit request

        # If n>1, call API directly with logprobs + stop support
        # (parent's generate_texts returns plain strings without logprobs)
        if n > 1:
            log.info(f"[IMPL] Batched generation with n={n} for {len(chats)} chat(s)")
            stop = args.get("stop", None)
            results = []
            for chat in chats:
                response = self.client.chat.completions.create(
                    model=self.model_path,
                    messages=chat,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    n=n,
                    stream=False,
                    logprobs=needs_logprobs,
                    top_logprobs=20 if needs_logprobs else None,
                    stop=stop,
                )
                chat_results = []
                for choice in response.choices:
                    text = choice.message.content or ""
                    logprobs_data = []
                    if needs_logprobs and choice.logprobs and choice.logprobs.content:
                        for token_info in choice.logprobs.content:
                            logprobs_data.append({
                                "token": token_info.token,
                                "logprob": token_info.logprob,
                                "top_logprobs": [
                                    {"token": t.token, "logprob": t.logprob}
                                    for t in token_info.top_logprobs
                                ],
                            })
                    chat_results.append({
                        "text": text,
                        "logprobs": logprobs_data,
                        "finish_reason": choice.finish_reason,
                    })
                results.append(chat_results)
            return results

        # Otherwise use streaming implementation (n=1)
        results = []
        for chat in chats:
            # Create streaming request
            stop = args.get("stop", None)
            response = self.client.chat.completions.create(
                model=self.model_path,
                messages=chat,
                max_tokens=max_new_tokens,
                temperature=temperature,
                stream=True,
                logprobs=needs_logprobs,
                top_logprobs=20 if needs_logprobs else None,
                stop=stop,
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

    def cleanup(self):
        """Clean up resources (shutdown thread pool executor)."""
        if hasattr(self, "_executor"):
            log.info("[CLEANUP] Shutting down thread pool executor")
            self._executor.shutdown(wait=True, cancel_futures=True)
            log.info("[CLEANUP] Executor shutdown complete")

    def __del__(self):
        """Destructor to ensure cleanup on object deletion."""
        try:
            self.cleanup()
        except Exception:
            pass  # Ignore errors during cleanup
