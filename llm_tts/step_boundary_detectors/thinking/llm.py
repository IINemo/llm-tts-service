"""
LLM-based step boundary detector for thinking mode.

Uses a secondary LLM to semantically parse thinking content into steps.
"""

import logging
import re
from typing import Any, Callable, List, Optional, Tuple

from ..base import StepBoundaryDetectorBase

log = logging.getLogger(__name__)

# Approximate chars per token (conservative estimate)
CHARS_PER_TOKEN = 4


# Default prompt for step extraction
DEFAULT_STEP_EXTRACTION_PROMPT = """You are a text segmentation expert. Your task is to split the following reasoning text into logical steps WITHOUT changing, summarizing, or omitting ANY words.

<thinking>
{thinking_content}
</thinking>

CRITICAL RULES:
1. Output EVERY SINGLE WORD from the input - do not skip, summarize, or paraphrase anything
2. Only add "Step N:" markers to indicate where each step begins
3. Each step should be a complete thought or reasoning unit
4. The concatenation of all steps must exactly reproduce the original text
5. Do not add any commentary, interpretation, or extra text

Output format - just add step markers before each logical segment:
Step 1: [exact original text of first reasoning step]
Step 2: [exact original text of second reasoning step]
...

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
        chunk_size: Optional[int] = None,
        chunk_overlap: int = 200,
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
            chunk_size: Max chars per chunk (auto-calculated if None based on max_tokens)
            chunk_overlap: Overlap between chunks in chars to avoid cutting steps
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
        self.chunk_overlap = chunk_overlap

        # Auto-calculate chunk size: smaller chunks = better preservation
        # For verbatim output, input and output are same size, so chunk must fit in context
        # GPT-4.1-mini has 128k context, but smaller chunks = more accurate preservation
        if chunk_size is None:
            # Use ~8k chars per chunk for better word preservation
            self.chunk_size = 8000
        else:
            self.chunk_size = chunk_size

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

        # Generate step extraction (with chunking for long content)
        try:
            if len(text) <= self.chunk_size:
                # Short content: process in one call
                llm_output = self._call_llm(text)
                steps = self._parse_llm_output(llm_output)
            else:
                # Long content: process in chunks
                steps = self._process_chunks(text)
        except Exception as e:
            log.error(f"LLM step detection failed: {e}")
            # Fallback: return entire text as single step
            steps = [text]

        # Cache result
        if self.cache_results:
            self._cache[cache_key] = steps

        return steps

    def _split_into_chunks(self, text: str) -> List[Tuple[int, str]]:
        """
        Split text into overlapping chunks at paragraph boundaries.

        Returns:
            List of (start_position, chunk_text) tuples
        """
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            if end >= len(text):
                # Last chunk
                chunks.append((start, text[start:]))
                break

            # Try to find a paragraph boundary near the end
            # Look for double newline within last 20% of chunk
            search_start = end - int(self.chunk_size * 0.2)
            search_region = text[search_start:end]

            # Find last paragraph break in search region
            para_break = search_region.rfind("\n\n")
            if para_break != -1:
                end = search_start + para_break + 2  # Include the newlines
            else:
                # No paragraph break, try single newline
                newline_break = search_region.rfind("\n")
                if newline_break != -1:
                    end = search_start + newline_break + 1

            chunks.append((start, text[start:end]))

            # Next chunk starts with overlap
            start = max(start + 1, end - self.chunk_overlap)

        return chunks

    def _process_chunks(self, text: str) -> List[str]:
        """
        Process long text in chunks and merge results.

        Returns:
            Merged list of steps from all chunks
        """
        chunks = self._split_into_chunks(text)
        log.info(
            f"Processing {len(chunks)} chunks for long content ({len(text)} chars)"
        )

        all_steps = []
        seen_steps = set()  # Track unique steps to handle overlaps

        for i, (start_pos, chunk) in enumerate(chunks):
            log.info(
                f"Processing chunk {i+1}/{len(chunks)} (start={start_pos}, len={len(chunk)} chars)"
            )

            try:
                llm_output = self._call_llm(chunk)
                log.info(
                    f"Chunk {i+1} LLM response ({len(llm_output)} chars):\n{llm_output}"
                )
                chunk_steps = self._parse_llm_output(llm_output)
                total_step_chars = sum(len(s) for s in chunk_steps)
                log.info(
                    f"Chunk {i+1}: got {len(chunk_steps)} steps, {total_step_chars} chars (input was {len(chunk)} chars, coverage {total_step_chars/len(chunk)*100:.1f}%)"
                )

                for step in chunk_steps:
                    # Use first 100 chars as dedup key to handle slight variations
                    step_key = step[:100].strip().lower()
                    if step_key not in seen_steps:
                        seen_steps.add(step_key)
                        all_steps.append(step)

            except Exception as e:
                log.warning(f"Chunk {i+1} failed: {e}")
                # Add chunk as single step on failure
                all_steps.append(chunk)

        return all_steps

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
