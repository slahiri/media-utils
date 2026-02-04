"""Tests for config module."""

import os
import pytest
from pathlib import Path

from media_utils.config import load_config, get_torch_dtype, DEFAULT_CONFIG_PATH


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_config_from_path(self, sample_config):
        """Test loading config from a specific path."""
        config = load_config(sample_config)

        assert "models" in config
        assert "paths" in config
        assert config["models"]["image"]["pipeline"]["name"] == "test/image-model"
        assert config["models"]["llm"]["name"] == "test/llm-model"

    def test_load_config_expands_paths(self, sample_config):
        """Test that paths with ~ are expanded."""
        config = load_config(sample_config)

        # Cache dir should not contain ~
        cache_dir = config["paths"]["cache_dir"]
        assert "~" not in cache_dir

    def test_load_config_missing_file(self, temp_dir):
        """Test loading config from non-existent path raises error."""
        with pytest.raises(FileNotFoundError):
            load_config(temp_dir / "nonexistent.yaml")

    def test_load_config_default_path_exists(self):
        """Test that default config path is set."""
        assert DEFAULT_CONFIG_PATH is not None
        assert DEFAULT_CONFIG_PATH.name == "config.yaml"

    def test_load_config_image_settings(self, sample_config):
        """Test image model configuration."""
        config = load_config(sample_config)
        image_config = config["models"]["image"]

        assert image_config["default_steps"] == 8
        assert image_config["default_size"] == [512, 512]
        assert image_config["guidance_scale"] == 1.0

    def test_load_config_split_files(self, sample_config):
        """Test split files configuration."""
        config = load_config(sample_config)
        split_config = config["models"]["image"]["split_files"]

        assert split_config["repo_id"] == "test/split-model"
        assert "text_encoder" in split_config
        assert "diffusion_model" in split_config
        assert "vae" in split_config

    def test_load_config_llm_settings(self, sample_config):
        """Test LLM configuration."""
        config = load_config(sample_config)
        llm_config = config["models"]["llm"]

        assert llm_config["name"] == "test/llm-model"
        assert llm_config["max_new_tokens"] == 100
        assert llm_config["torch_dtype"] == "float16"


class TestGetTorchDtype:
    """Tests for get_torch_dtype function."""

    def test_get_torch_dtype_float32(self):
        """Test converting float32 string."""
        import torch
        dtype = get_torch_dtype("float32")
        assert dtype == torch.float32

    def test_get_torch_dtype_float16(self):
        """Test converting float16 string."""
        import torch
        dtype = get_torch_dtype("float16")
        assert dtype == torch.float16

    def test_get_torch_dtype_bfloat16(self):
        """Test converting bfloat16 string."""
        import torch
        dtype = get_torch_dtype("bfloat16")
        assert dtype == torch.bfloat16

    def test_get_torch_dtype_unknown_returns_bfloat16(self):
        """Test that unknown dtype returns bfloat16 as default."""
        import torch
        dtype = get_torch_dtype("unknown")
        assert dtype == torch.bfloat16
