import os
from pathlib import Path
from typing import Any

from media_utils.config import load_config, get_torch_dtype


class QwenLLM:
    """Qwen LLM wrapper using HuggingFace transformers."""

    def __init__(
        self,
        config_path: str | Path | None = None,
        model_name: str | None = None,
        local_path: str | Path | None = None,
        device: str | None = None,
        torch_dtype: str | None = None,
    ):
        """Initialize the Qwen LLM.

        Args:
            config_path: Path to config file. Uses default if not provided.
            model_name: Override model name from config.
            local_path: Load from local path instead of HuggingFace.
            device: Override device from config.
            torch_dtype: Override torch dtype from config.
        """
        self.config = load_config(config_path)
        self.config_path = config_path
        llm_config = self.config["models"]["llm"]

        self.model_name = model_name or llm_config["name"]
        self.local_path = local_path
        self.device = device or llm_config["device"]
        self.torch_dtype = get_torch_dtype(torch_dtype or llm_config["torch_dtype"])
        self.max_new_tokens = llm_config.get("max_new_tokens", 512)

        # Check for local path in config if not provided
        if self.local_path is None and "local_path" in llm_config:
            project_root = self._get_project_root()
            potential_local = project_root / llm_config["local_path"]
            if potential_local.exists():
                self.local_path = str(potential_local)

        self._model = None
        self._tokenizer = None

    def _get_project_root(self) -> Path:
        """Get the project root directory."""
        if self.config_path:
            return Path(self.config_path).parent
        return Path(__file__).parent.parent.parent

    @property
    def model(self):
        """Lazy load the model."""
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def tokenizer(self):
        """Lazy load the tokenizer."""
        if self._tokenizer is None:
            self._load_model()
        return self._tokenizer

    def _load_model(self):
        """Load the model and tokenizer."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cache_dir = self.config.get("paths", {}).get("cache_dir")
        if cache_dir:
            cache_dir = os.path.expanduser(cache_dir)

        # Use local path if available, otherwise use HuggingFace
        model_path = self.local_path if self.local_path else self.model_name

        print(f"Loading LLM from: {model_path}")

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            cache_dir=cache_dir if not self.local_path else None,
        )

        self._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=self.torch_dtype,
            device_map=self.device,
            cache_dir=cache_dir if not self.local_path else None,
        )

    def generate(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs: Any,
    ) -> str:
        """Generate text from a prompt.

        Args:
            prompt: Input text prompt.
            max_new_tokens: Maximum tokens to generate. Defaults to config value.
            temperature: Sampling temperature.
            top_p: Top-p sampling parameter.
            **kwargs: Additional arguments passed to model.generate().

        Returns:
            Generated text string.
        """
        max_new_tokens = max_new_tokens or self.max_new_tokens

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            **kwargs,
        )

        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    def chat(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int | None = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs: Any,
    ) -> str:
        """Generate a response in chat format.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
                     Roles can be 'system', 'user', or 'assistant'.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Top-p sampling parameter.
            **kwargs: Additional arguments passed to model.generate().

        Returns:
            Assistant's response text.
        """
        max_new_tokens = max_new_tokens or self.max_new_tokens

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            **kwargs,
        )

        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
