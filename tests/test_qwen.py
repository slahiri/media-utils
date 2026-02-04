"""Tests for QwenLLM class."""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from media_utils.llm.qwen import QwenLLM


class TestQwenLLMInit:
    """Tests for QwenLLM initialization."""

    def test_init_with_config(self, sample_config):
        """Test initialization with config path."""
        llm = QwenLLM(config_path=sample_config)

        assert llm.model_name == "test/llm-model"
        assert llm.device == "cpu"
        assert llm.max_new_tokens == 100

    def test_init_with_model_name_override(self, sample_config):
        """Test initialization with model name override."""
        llm = QwenLLM(config_path=sample_config, model_name="custom/llm")
        assert llm.model_name == "custom/llm"

    def test_init_with_device_override(self, sample_config):
        """Test initialization with device override."""
        llm = QwenLLM(config_path=sample_config, device="cuda:0")
        assert llm.device == "cuda:0"

    def test_init_with_local_path(self, sample_config):
        """Test initialization with explicit local path."""
        llm = QwenLLM(config_path=sample_config, local_path="/custom/local/path")
        assert llm.local_path == "/custom/local/path"

    def test_init_lazy_load(self, sample_config):
        """Test that model is not loaded on init."""
        llm = QwenLLM(config_path=sample_config)
        assert llm._model is None
        assert llm._tokenizer is None


class TestQwenLLMModelLoading:
    """Tests for QwenLLM model loading."""

    def test_model_property_triggers_load(self, sample_config):
        """Test that accessing model property triggers loading."""
        llm = QwenLLM(config_path=sample_config)

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        with patch("transformers.AutoModelForCausalLM") as mock_model_class, \
             patch("transformers.AutoTokenizer") as mock_tokenizer_class:
            mock_model_class.from_pretrained.return_value = mock_model
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

            # Access model property
            _ = llm.model

            # Model should be loaded
            mock_model_class.from_pretrained.assert_called_once()

    def test_tokenizer_property_triggers_load(self, sample_config):
        """Test that accessing tokenizer property triggers loading."""
        llm = QwenLLM(config_path=sample_config)

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        with patch("transformers.AutoModelForCausalLM") as mock_model_class, \
             patch("transformers.AutoTokenizer") as mock_tokenizer_class:
            mock_model_class.from_pretrained.return_value = mock_model
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

            # Access tokenizer property
            _ = llm.tokenizer

            # Tokenizer should be loaded
            mock_tokenizer_class.from_pretrained.assert_called_once()


class TestQwenLLMGenerate:
    """Tests for QwenLLM.generate method."""

    def _setup_mocks(self, llm):
        """Setup mock model and tokenizer for testing."""
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()

        # Mock tokenizer call returns a mock that can be .to()'d
        mock_inputs = MagicMock()
        mock_inputs.__getitem__ = lambda self, key: MagicMock(shape=[1, 10])
        mock_inputs.to.return_value = mock_inputs
        mock_tokenizer.return_value = mock_inputs

        mock_tokenizer.eos_token_id = 0
        mock_tokenizer.decode.return_value = "Generated response"

        # Mock model
        mock_model.device = "cpu"
        mock_output_tensor = MagicMock()
        mock_output_tensor.__getitem__ = lambda self, idx: MagicMock()
        mock_model.generate.return_value = [mock_output_tensor]

        llm._model = mock_model
        llm._tokenizer = mock_tokenizer

        return mock_model, mock_tokenizer

    def test_generate_basic(self, sample_config):
        """Test basic text generation."""
        llm = QwenLLM(config_path=sample_config)
        mock_model, mock_tokenizer = self._setup_mocks(llm)

        result = llm.generate("Test prompt")

        assert result == "Generated response"
        mock_model.generate.assert_called_once()

    def test_generate_with_custom_params(self, sample_config):
        """Test generation with custom parameters."""
        llm = QwenLLM(config_path=sample_config)
        mock_model, mock_tokenizer = self._setup_mocks(llm)

        result = llm.generate(
            "Test prompt",
            max_new_tokens=200,
            temperature=0.5,
            top_p=0.8,
        )

        call_kwargs = mock_model.generate.call_args[1]
        assert call_kwargs["max_new_tokens"] == 200
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["top_p"] == 0.8

    def test_generate_uses_config_defaults(self, sample_config):
        """Test that generate uses config max_new_tokens default."""
        llm = QwenLLM(config_path=sample_config)
        mock_model, mock_tokenizer = self._setup_mocks(llm)

        result = llm.generate("Test prompt")

        call_kwargs = mock_model.generate.call_args[1]
        assert call_kwargs["max_new_tokens"] == 100  # From config


class TestQwenLLMChat:
    """Tests for QwenLLM.chat method."""

    def _setup_mocks(self, llm):
        """Setup mock model and tokenizer for testing."""
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()

        # Mock chat template
        mock_tokenizer.apply_chat_template.return_value = "Formatted chat"

        # Mock tokenizer call returns a mock that can be .to()'d
        mock_inputs = MagicMock()
        mock_inputs.__getitem__ = lambda self, key: MagicMock(shape=[1, 10])
        mock_inputs.to.return_value = mock_inputs
        mock_tokenizer.return_value = mock_inputs

        mock_tokenizer.eos_token_id = 0
        mock_tokenizer.decode.return_value = "Chat response"

        # Mock model
        mock_model.device = "cpu"
        mock_output_tensor = MagicMock()
        mock_output_tensor.__getitem__ = lambda self, idx: MagicMock()
        mock_model.generate.return_value = [mock_output_tensor]

        llm._model = mock_model
        llm._tokenizer = mock_tokenizer

        return mock_model, mock_tokenizer

    def test_chat_basic(self, sample_config):
        """Test basic chat generation."""
        llm = QwenLLM(config_path=sample_config)
        mock_model, mock_tokenizer = self._setup_mocks(llm)

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
        ]

        result = llm.chat(messages)

        assert result == "Chat response"
        mock_tokenizer.apply_chat_template.assert_called_once_with(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def test_chat_with_custom_params(self, sample_config):
        """Test chat with custom parameters."""
        llm = QwenLLM(config_path=sample_config)
        mock_model, mock_tokenizer = self._setup_mocks(llm)

        messages = [{"role": "user", "content": "Hi"}]

        result = llm.chat(
            messages,
            max_new_tokens=50,
            temperature=0.3,
        )

        call_kwargs = mock_model.generate.call_args[1]
        assert call_kwargs["max_new_tokens"] == 50
        assert call_kwargs["temperature"] == 0.3
