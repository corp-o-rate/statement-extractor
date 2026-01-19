"""
LLM module for text generation using local models.

Supports:
- GGUF models via llama-cpp-python (efficient quantized inference)
- Transformers models via HuggingFace

Usage:
    from statement_extractor.llm import LLM

    llm = LLM()  # Uses default Gemma3 12B GGUF
    response = llm.generate("Your prompt here")
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class LLM:
    """
    LLM wrapper for text generation.

    Automatically selects the best backend:
    - GGUF models use llama-cpp-python (efficient, no de-quantization)
    - Other models use HuggingFace transformers
    """

    def __init__(
        self,
        model_id: str = "google/gemma-3-12b-it-qat-q4_0-gguf",
        gguf_file: Optional[str] = None,
        n_ctx: int = 8192,
        use_4bit: bool = True,
    ):
        """
        Initialize the LLM.

        Args:
            model_id: HuggingFace model ID
            gguf_file: GGUF filename (auto-detected if model_id ends with -gguf)
            n_ctx: Context size for GGUF models
            use_4bit: Use 4-bit quantization for transformers models
        """
        self._model_id = model_id
        self._gguf_file = gguf_file
        self._n_ctx = n_ctx
        self._use_4bit = use_4bit

        # Model instances (lazy loaded)
        self._llama_model = None  # llama-cpp-python
        self._transformers_model = None  # HuggingFace transformers
        self._tokenizer = None

        self._load_failed = False

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._llama_model is not None or self._transformers_model is not None

    def _is_gguf_model(self) -> bool:
        """Check if the model ID is a GGUF model."""
        return self._model_id.endswith("-gguf") or self._gguf_file is not None

    def _get_gguf_filename(self) -> str:
        """Get the GGUF filename from the model ID."""
        if self._gguf_file:
            return self._gguf_file
        # Extract filename from model ID like "google/gemma-3-12b-it-qat-q4_0-gguf"
        # The actual file is "gemma-3-12b-it-q4_0.gguf" (note: "qat" is removed)
        model_name = self._model_id.split("/")[-1]
        if model_name.endswith("-gguf"):
            model_name = model_name[:-5]  # Remove "-gguf" suffix
        # Remove "-qat" from the name (it's not in the actual filename)
        model_name = model_name.replace("-qat", "")
        return model_name + ".gguf"

    def load(self) -> None:
        """
        Load the model.

        Raises:
            RuntimeError: If the model fails to load
        """
        if self.is_loaded or self._load_failed:
            return

        try:
            logger.debug(f"Loading LLM: {self._model_id}")

            if self._is_gguf_model():
                self._load_gguf_model()
            else:
                self._load_transformers_model()

            logger.debug("LLM loaded successfully")

        except Exception as e:
            self._load_failed = True
            error_msg = f"Failed to load LLM ({self._model_id}): {e}"
            if "llama_cpp" in str(e).lower() or "llama-cpp" in str(e).lower():
                error_msg += "\n  Install with: pip install llama-cpp-python"
            if "accelerate" in str(e):
                error_msg += "\n  Install with: pip install accelerate"
            raise RuntimeError(error_msg) from e

    def _load_gguf_model(self) -> None:
        """Load GGUF model using llama-cpp-python."""
        try:
            from llama_cpp import Llama
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise ImportError(
                "llama-cpp-python is required for GGUF models. "
                "Install with: pip install llama-cpp-python"
            ) from e

        gguf_file = self._get_gguf_filename()
        logger.debug(f"Loading GGUF model with file: {gguf_file}")

        # Download the GGUF file from HuggingFace
        model_path = hf_hub_download(
            repo_id=self._model_id,
            filename=gguf_file,
        )

        # Load with llama-cpp-python
        self._llama_model = Llama(
            model_path=model_path,
            n_ctx=self._n_ctx,
            n_gpu_layers=-1,  # Use all GPU layers (Metal on Mac, CUDA on Linux)
            verbose=False,
        )

    def _load_transformers_model(self) -> None:
        """Load model using HuggingFace transformers."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)

        if self._use_4bit:
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
                self._transformers_model = AutoModelForCausalLM.from_pretrained(
                    self._model_id,
                    quantization_config=quantization_config,
                    device_map="auto",
                )
            except ImportError:
                logger.debug("bitsandbytes not available, loading full precision")
                self._transformers_model = AutoModelForCausalLM.from_pretrained(
                    self._model_id,
                    device_map="auto",
                    torch_dtype=torch.float16,
                )
        else:
            self._transformers_model = AutoModelForCausalLM.from_pretrained(
                self._model_id,
                device_map="auto",
                torch_dtype=torch.float16,
            )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        stop: Optional[list[str]] = None,
    ) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: The input prompt
            max_tokens: Maximum tokens to generate
            stop: Stop sequences

        Returns:
            Generated text (not including the prompt)
        """
        self.load()

        if self._llama_model is not None:
            return self._generate_with_llama(prompt, max_tokens, stop)
        else:
            return self._generate_with_transformers(prompt, max_tokens)

    def _generate_with_llama(
        self,
        prompt: str,
        max_tokens: int,
        stop: Optional[list[str]],
    ) -> str:
        """Generate response using llama-cpp-python."""
        output = self._llama_model(
            prompt,
            max_tokens=max_tokens,
            stop=stop or ["\n\n", "</s>"],
            echo=False,
        )
        return output["choices"][0]["text"]

    def _generate_with_transformers(
        self,
        prompt: str,
        max_tokens: int,
    ) -> str:
        """Generate response using transformers."""
        import torch

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._transformers_model.device)

        with torch.no_grad():
            outputs = self._transformers_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        return self._tokenizer.decode(outputs[0], skip_special_tokens=True)


# Singleton instance for shared use
_default_llm: Optional[LLM] = None


def get_llm(
    model_id: str = "google/gemma-3-12b-it-qat-q4_0-gguf",
    **kwargs,
) -> LLM:
    """
    Get or create a shared LLM instance.

    Uses a singleton pattern to avoid loading the model multiple times.

    Args:
        model_id: HuggingFace model ID
        **kwargs: Additional arguments passed to LLM constructor

    Returns:
        LLM instance
    """
    global _default_llm

    if _default_llm is None or _default_llm._model_id != model_id:
        _default_llm = LLM(model_id=model_id, **kwargs)

    return _default_llm
