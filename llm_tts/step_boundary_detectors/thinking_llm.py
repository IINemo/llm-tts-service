"""
LLM-based step boundary detector for thinking mode.

Uses a secondary LLM to semantically parse thinking content into steps.
"""

import logging
import re
from typing import Any, Callable, List, Optional

from .base import StepBoundaryDetectorBase

log = logging.getLogger(__name__)


# Default prompt for step extraction
DEFAULT_STEP_EXTRACTION_PROMPT = """You are an expert at analyzing reasoning processes. Given the following thinking/reasoning text, identify and extract the distinct reasoning steps.

<thinking>
{thinking_content}
</thinking>

Extract each logical reasoning step. Output ONLY the steps, one per line, starting each with "Step N:" where N is the step number.

Rules:
1. Each step should be a complete thought or reasoning unit
2. Preserve the original wording as much as possible
3. Don't add interpretation or explanation
4. If the text has no clear steps, output it as a single "Step 1:"

Output:"""


class ThinkingLLMDetector(StepBoundaryDetectorBase):
    """
    Detects step boundaries in <think> content using an LLM.

    Sends thinking content to a secondary LLM (can be smaller/faster)
    which parses it into logical reasoning steps.

    Best for: High-quality semantic step detection when compute budget allows.
    """

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        llm_generate_fn: Optional[Callable[[str], str]] = None,
        model_name: str = "gpt-4o-mini",
        prompt_template: Optional[str] = None,
        api_base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        cache_results: bool = True,
    ):
        """
        Args:
            llm_client: Pre-configured LLM client (OpenAI, vLLM, etc.)
            llm_generate_fn: Custom generation function (text) -> str
            model_name: Model to use if creating client
            prompt_template: Custom prompt template with {thinking_content} placeholder
            api_base_url: API base URL (for OpenAI-compatible endpoints)
            api_key: API key (if not using environment variable)
            temperature: Sampling temperature (0 for deterministic)
            max_tokens: Maximum tokens in response
            cache_results: Whether to cache results for identical inputs
        """
        self.llm_client = llm_client
        self.llm_generate_fn = llm_generate_fn
        self.model_name = model_name
        self.prompt_template = prompt_template or DEFAULT_STEP_EXTRACTION_PROMPT
        self.api_base_url = api_base_url
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cache_results = cache_results

        # Simple in-memory cache
        self._cache: dict = {}

        # Initialize client if needed
        if not self.llm_client and not self.llm_generate_fn:
            self._init_default_client()

    def _init_default_client(self):
        """Initialize default OpenAI client."""
        try:
            import openai

            client_kwargs = {}
            if self.api_base_url:
                client_kwargs["base_url"] = self.api_base_url
            if self.api_key:
                client_kwargs["api_key"] = self.api_key

            self.llm_client = openai.OpenAI(**client_kwargs)
            log.info(f"Initialized OpenAI client for model: {self.model_name}")
        except ImportError:
            log.warning(
                "OpenAI not installed. Provide llm_client or llm_generate_fn, "
                "or install openai: pip install openai"
            )

    def detect_steps(self, text: str, **kwargs) -> List[str]:
        """
        Detect steps using LLM parsing.

        Args:
            text: Thinking content (inside <think> tags)

        Returns:
            List of step strings
        """
        text = self._extract_thinking_content(text)

        if not text.strip():
            return []

        # Check cache
        if self.cache_results:
            cache_key = hash(text)
            if cache_key in self._cache:
                log.debug("Using cached step detection result")
                return self._cache[cache_key]

        # Generate step extraction
        try:
            llm_output = self._call_llm(text)
            steps = self._parse_llm_output(llm_output)
        except Exception as e:
            log.error(f"LLM step detection failed: {e}")
            # Fallback: return entire text as single step
            steps = [text]

        # Cache result
        if self.cache_results:
            self._cache[cache_key] = steps

        return steps

    def _extract_thinking_content(self, text: str) -> str:
        """Extract content from <think> tags if present."""
        think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        if think_match:
            return think_match.group(1).strip()
        return text.strip()

    def _call_llm(self, thinking_content: str) -> str:
        """Call the LLM to parse thinking content."""
        prompt = self.prompt_template.format(thinking_content=thinking_content)

        # Use custom generation function if provided
        if self.llm_generate_fn:
            return self.llm_generate_fn(prompt)

        # Use OpenAI-style client
        if self.llm_client:
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content

        raise RuntimeError("No LLM client or generation function configured")

    def _parse_llm_output(self, output: str) -> List[str]:
        """Parse LLM output into list of steps."""
        steps = []

        # Match "Step N:" patterns
        step_pattern = r"Step\s*\d+:\s*(.*?)(?=Step\s*\d+:|$)"
        matches = re.findall(step_pattern, output, re.DOTALL | re.IGNORECASE)

        if matches:
            for match in matches:
                step_text = match.strip()
                if step_text:
                    steps.append(step_text)
        else:
            # Fallback: split by newlines if no step markers found
            lines = [line.strip() for line in output.split("\n") if line.strip()]
            steps = lines if lines else [output]

        return steps

    def clear_cache(self):
        """Clear the result cache."""
        self._cache.clear()

    def set_prompt_template(self, template: str):
        """
        Set a custom prompt template.

        Template must include {thinking_content} placeholder.
        """
        if "{thinking_content}" not in template:
            raise ValueError("Template must include {thinking_content} placeholder")
        self.prompt_template = template


class ThinkingLLMDetectorVLLM(ThinkingLLMDetector):
    """
    LLM-based detector using vLLM for fast local inference.

    Optimized for running alongside the main model evaluation.
    """

    def __init__(
        self,
        vllm_engine: Optional[Any] = None,
        model_path: str = "Qwen/Qwen2.5-0.5B-Instruct",
        prompt_template: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        cache_results: bool = True,
        gpu_memory_utilization: float = 0.1,
    ):
        """
        Args:
            vllm_engine: Pre-initialized vLLM engine
            model_path: Path to model (used if vllm_engine not provided)
            prompt_template: Custom prompt template
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            cache_results: Whether to cache results
            gpu_memory_utilization: GPU memory fraction for small model
        """
        self.vllm_engine = vllm_engine
        self.model_path = model_path
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.gpu_memory_utilization = gpu_memory_utilization

        # Initialize parent without default client
        super().__init__(
            llm_client=None,
            llm_generate_fn=self._vllm_generate,
            prompt_template=prompt_template,
            temperature=temperature,
            max_tokens=max_tokens,
            cache_results=cache_results,
        )

    def _vllm_generate(self, prompt: str) -> str:
        """Generate using vLLM engine."""
        if self.vllm_engine is None:
            self._init_vllm_engine()

        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        outputs = self.vllm_engine.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text

    def _init_vllm_engine(self):
        """Initialize vLLM engine with small model."""
        try:
            from vllm import LLM

            log.info(f"Initializing vLLM engine for step detection: {self.model_path}")
            self.vllm_engine = LLM(
                model=self.model_path,
                gpu_memory_utilization=self.gpu_memory_utilization,
                trust_remote_code=True,
            )
        except ImportError:
            raise ImportError("vLLM not installed. Install with: pip install vllm")
