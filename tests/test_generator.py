"""Tests for ImageGenerator class."""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from pathlib import Path

from media_utils.image.generator import ImageGenerator


class TestImageGeneratorInit:
    """Tests for ImageGenerator initialization."""

    def test_init_with_default_config(self, sample_config):
        """Test initialization with config path."""
        gen = ImageGenerator(config_path=sample_config)

        assert gen.mode == "pipeline"
        assert gen.model_name == "test/image-model"
        assert gen.device == "cpu"
        assert gen.default_steps == 8
        assert gen.default_size == [512, 512]

    def test_init_with_mode_override(self, sample_config):
        """Test initialization with mode override."""
        gen = ImageGenerator(config_path=sample_config, mode="local")
        assert gen.mode == "local"

    def test_init_with_device_override(self, sample_config):
        """Test initialization with device override."""
        gen = ImageGenerator(config_path=sample_config, device="cuda:1")
        assert gen.device == "cuda:1"

    def test_init_with_model_name_override(self, sample_config):
        """Test initialization with model name override."""
        gen = ImageGenerator(config_path=sample_config, model_name="custom/model")
        assert gen.model_name == "custom/model"

    def test_init_offload_mode(self, sample_config):
        """Test offload mode setting."""
        gen = ImageGenerator(config_path=sample_config, offload_mode="model")
        assert gen.offload_mode == "model"


class TestImageGeneratorPipeline:
    """Tests for ImageGenerator pipeline loading."""

    def test_pipeline_lazy_load(self, sample_config):
        """Test that pipeline is not loaded on init."""
        gen = ImageGenerator(config_path=sample_config)
        assert gen._pipeline is None

    @patch("media_utils.image.generator.ZImagePipeline", create=True)
    def test_pipeline_loads_on_access(self, mock_pipeline_class, sample_config):
        """Test that pipeline loads when accessed."""
        # Setup mock
        mock_pipeline = MagicMock()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline

        with patch.dict("sys.modules", {"diffusers": MagicMock(ZImagePipeline=mock_pipeline_class)}):
            gen = ImageGenerator(config_path=sample_config)

            # Patch the import inside _load_pipeline
            with patch("media_utils.image.generator.ZImagePipeline", mock_pipeline_class, create=True):
                # Access pipeline property - this would trigger loading
                # We can't fully test this without the actual diffusers module
                pass


class TestImageGeneratorGenerate:
    """Tests for ImageGenerator.generate method."""

    def test_generate_uses_defaults(self, sample_config):
        """Test that generate uses config defaults."""
        gen = ImageGenerator(config_path=sample_config)

        # Mock the pipeline
        mock_pipeline = MagicMock()
        mock_image = MagicMock()
        mock_pipeline.return_value.images = [mock_image]
        gen._pipeline = mock_pipeline

        result = gen.generate("test prompt")

        # Check pipeline was called with defaults
        call_kwargs = mock_pipeline.call_args[1]
        assert call_kwargs["height"] == 512
        assert call_kwargs["width"] == 512
        assert call_kwargs["num_inference_steps"] == 8
        assert call_kwargs["guidance_scale"] == 1.0

    def test_generate_with_custom_params(self, sample_config):
        """Test generate with custom parameters."""
        gen = ImageGenerator(config_path=sample_config)

        mock_pipeline = MagicMock()
        mock_image = MagicMock()
        mock_pipeline.return_value.images = [mock_image]
        gen._pipeline = mock_pipeline

        result = gen.generate(
            "test prompt",
            height=1024,
            width=1024,
            num_inference_steps=20,
            guidance_scale=2.0,
        )

        call_kwargs = mock_pipeline.call_args[1]
        assert call_kwargs["height"] == 1024
        assert call_kwargs["width"] == 1024
        assert call_kwargs["num_inference_steps"] == 20
        assert call_kwargs["guidance_scale"] == 2.0

    def test_generate_with_seed(self, sample_config):
        """Test generate with seed for reproducibility."""
        gen = ImageGenerator(config_path=sample_config)
        gen.device = "cpu"  # Use CPU for test

        mock_pipeline = MagicMock()
        mock_image = MagicMock()
        mock_pipeline.return_value.images = [mock_image]
        gen._pipeline = mock_pipeline

        with patch("torch.Generator") as mock_generator:
            mock_gen_instance = MagicMock()
            mock_generator.return_value = mock_gen_instance
            mock_gen_instance.manual_seed.return_value = mock_gen_instance

            result = gen.generate("test prompt", seed=42)

            mock_generator.assert_called_once_with("cpu")
            mock_gen_instance.manual_seed.assert_called_once_with(42)

    def test_generate_returns_image(self, sample_config):
        """Test that generate returns an image."""
        gen = ImageGenerator(config_path=sample_config)

        mock_pipeline = MagicMock()
        mock_image = MagicMock()
        mock_pipeline.return_value.images = [mock_image]
        gen._pipeline = mock_pipeline

        result = gen.generate("test prompt")

        assert result == mock_image


class TestImageGeneratorBatch:
    """Tests for ImageGenerator.generate_batch method."""

    def test_generate_batch(self, sample_config):
        """Test batch generation."""
        gen = ImageGenerator(config_path=sample_config)

        mock_pipeline = MagicMock()
        mock_images = [MagicMock(), MagicMock(), MagicMock()]
        call_count = [0]

        def side_effect(*args, **kwargs):
            result = MagicMock()
            result.images = [mock_images[call_count[0]]]
            call_count[0] += 1
            return result

        mock_pipeline.side_effect = side_effect
        gen._pipeline = mock_pipeline

        prompts = ["prompt 1", "prompt 2", "prompt 3"]
        results = gen.generate_batch(prompts)

        assert len(results) == 3
        assert mock_pipeline.call_count == 3


class TestImageGeneratorLocalMode:
    """Tests for ImageGenerator local mode."""

    def test_check_local_models_exist_false(self, sample_config):
        """Test checking for non-existent local models."""
        gen = ImageGenerator(config_path=sample_config, mode="local")

        assert gen._check_local_models_exist() is False

    def test_check_local_models_exist_true(self, sample_config, temp_dir):
        """Test checking for existing local models."""
        # Create fake model files
        (temp_dir / "models" / "text_encoders" / "test.safetensors").touch()
        (temp_dir / "models" / "diffusion_models" / "test.safetensors").touch()
        (temp_dir / "models" / "vae" / "test.safetensors").touch()

        gen = ImageGenerator(config_path=sample_config, mode="local")

        assert gen._check_local_models_exist() is True

    def test_get_local_model_paths(self, sample_config, temp_dir):
        """Test getting local model paths."""
        gen = ImageGenerator(config_path=sample_config, mode="local")

        paths = gen._get_local_model_paths()

        assert "text_encoder" in paths
        assert "diffusion_model" in paths
        assert "vae" in paths
        assert paths["text_encoder"].name == "test.safetensors"
