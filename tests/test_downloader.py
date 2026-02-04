"""Tests for downloader module."""

import pytest
from unittest.mock import MagicMock, patch, call
from pathlib import Path

from media_utils.utils.downloader import (
    download_image_model,
    download_image_model_pipeline,
    download_image_model_split,
    download_llm_model,
    download_models,
    get_project_root,
    list_available_models,
)


class TestGetProjectRoot:
    """Tests for get_project_root function."""

    def test_get_project_root_with_config_path(self, sample_config):
        """Test getting project root from config path."""
        root = get_project_root(sample_config)
        assert root == sample_config.parent

    def test_get_project_root_finds_config_yaml(self, temp_dir):
        """Test finding project root by config.yaml location."""
        # This test depends on the actual project structure
        root = get_project_root()
        assert root.is_dir()


class TestDownloadImageModelPipeline:
    """Tests for download_image_model_pipeline function."""

    def test_download_pipeline_basic(self, sample_config, mock_huggingface_hub):
        """Test basic pipeline download."""
        with patch("media_utils.utils.downloader.snapshot_download", mock_huggingface_hub["snapshot_download"]):
            result = download_image_model_pipeline(sample_config)

            assert result == "/fake/path/to/model"
            mock_huggingface_hub["snapshot_download"].assert_called_once()

    def test_download_pipeline_uses_config_model(self, sample_config, mock_huggingface_hub):
        """Test that pipeline download uses model from config."""
        with patch("media_utils.utils.downloader.snapshot_download", mock_huggingface_hub["snapshot_download"]):
            download_image_model_pipeline(sample_config)

            call_kwargs = mock_huggingface_hub["snapshot_download"].call_args
            assert call_kwargs[1]["repo_id"] == "test/image-model"

    def test_download_pipeline_with_model_override(self, sample_config, mock_huggingface_hub):
        """Test pipeline download with model name override."""
        with patch("media_utils.utils.downloader.snapshot_download", mock_huggingface_hub["snapshot_download"]):
            download_image_model_pipeline(sample_config, model_name="custom/model")

            call_kwargs = mock_huggingface_hub["snapshot_download"].call_args
            assert call_kwargs[1]["repo_id"] == "custom/model"


class TestDownloadImageModelSplit:
    """Tests for download_image_model_split function."""

    def test_download_split_downloads_all_files(self, sample_config, mock_huggingface_hub):
        """Test that split download fetches all component files."""
        with patch("media_utils.utils.downloader.hf_hub_download", mock_huggingface_hub["hf_hub_download"]):
            result = download_image_model_split(sample_config)

            # Should download 3 files: text_encoder, diffusion_model, vae
            assert mock_huggingface_hub["hf_hub_download"].call_count == 3
            assert "text_encoder" in result
            assert "diffusion_model" in result
            assert "vae" in result

    def test_download_split_uses_correct_repo(self, sample_config, mock_huggingface_hub):
        """Test that split download uses correct repo ID."""
        with patch("media_utils.utils.downloader.hf_hub_download", mock_huggingface_hub["hf_hub_download"]):
            download_image_model_split(sample_config)

            # Check all calls use the correct repo
            for call_args in mock_huggingface_hub["hf_hub_download"].call_args_list:
                assert call_args[1]["repo_id"] == "test/split-model"

    def test_download_split_copy_to_local(self, sample_config, mock_huggingface_hub, temp_dir):
        """Test copying downloaded files to local folder."""
        # Create a fake cached file to copy
        fake_cached_file = temp_dir / "cached_file.safetensors"
        fake_cached_file.write_text("fake model data")

        mock_huggingface_hub["hf_hub_download"].return_value = str(fake_cached_file)

        with patch("media_utils.utils.downloader.hf_hub_download", mock_huggingface_hub["hf_hub_download"]):
            result = download_image_model_split(sample_config, copy_to_local=True)

            # Should have local paths in result
            assert "text_encoder_local" in result
            assert "diffusion_model_local" in result
            assert "vae_local" in result


class TestDownloadImageModel:
    """Tests for download_image_model function."""

    def test_download_image_model_pipeline_mode(self, sample_config, mock_huggingface_hub):
        """Test download_image_model with pipeline mode."""
        with patch("media_utils.utils.downloader.snapshot_download", mock_huggingface_hub["snapshot_download"]):
            result = download_image_model(sample_config, mode="pipeline")

            assert result == "/fake/path/to/model"

    def test_download_image_model_split_mode(self, sample_config, mock_huggingface_hub):
        """Test download_image_model with split mode."""
        with patch("media_utils.utils.downloader.hf_hub_download", mock_huggingface_hub["hf_hub_download"]):
            result = download_image_model(sample_config, mode="split")

            assert isinstance(result, dict)

    def test_download_image_model_invalid_mode(self, sample_config):
        """Test download_image_model with invalid mode raises error."""
        with pytest.raises(ValueError, match="Unknown mode"):
            download_image_model(sample_config, mode="invalid")


class TestDownloadLLMModel:
    """Tests for download_llm_model function."""

    def test_download_llm_basic(self, sample_config, mock_huggingface_hub):
        """Test basic LLM download."""
        with patch("media_utils.utils.downloader.snapshot_download", mock_huggingface_hub["snapshot_download"]):
            result = download_llm_model(sample_config)

            assert result == "/fake/path/to/model"
            mock_huggingface_hub["snapshot_download"].assert_called_once()

    def test_download_llm_uses_config_model(self, sample_config, mock_huggingface_hub):
        """Test that LLM download uses model from config."""
        with patch("media_utils.utils.downloader.snapshot_download", mock_huggingface_hub["snapshot_download"]):
            download_llm_model(sample_config)

            call_kwargs = mock_huggingface_hub["snapshot_download"].call_args
            assert call_kwargs[1]["repo_id"] == "test/llm-model"

    def test_download_llm_with_model_override(self, sample_config, mock_huggingface_hub):
        """Test LLM download with model name override."""
        with patch("media_utils.utils.downloader.snapshot_download", mock_huggingface_hub["snapshot_download"]):
            download_llm_model(sample_config, model_name="custom/llm")

            call_kwargs = mock_huggingface_hub["snapshot_download"].call_args
            assert call_kwargs[1]["repo_id"] == "custom/llm"


class TestDownloadModels:
    """Tests for download_models function."""

    def test_download_models_downloads_both(self, sample_config, mock_huggingface_hub):
        """Test that download_models downloads both image and LLM models."""
        with patch("media_utils.utils.downloader.snapshot_download", mock_huggingface_hub["snapshot_download"]):
            result = download_models(sample_config, image_mode="pipeline")

            assert "image" in result
            assert "llm" in result
            # Called twice: once for image, once for LLM
            assert mock_huggingface_hub["snapshot_download"].call_count == 2

    def test_download_models_split_mode(self, sample_config, mock_huggingface_hub):
        """Test download_models with split mode for image."""
        with patch("media_utils.utils.downloader.snapshot_download", mock_huggingface_hub["snapshot_download"]), \
             patch("media_utils.utils.downloader.hf_hub_download", mock_huggingface_hub["hf_hub_download"]):
            result = download_models(sample_config, image_mode="split")

            assert isinstance(result["image"], dict)  # Split returns dict
            assert isinstance(result["llm"], str)  # LLM returns path string


class TestListAvailableModels:
    """Tests for list_available_models function."""

    def test_list_available_models_runs(self, capsys):
        """Test that list_available_models prints output."""
        list_available_models()

        captured = capsys.readouterr()
        assert "Z-Image-Turbo" in captured.out
        assert "Qwen" in captured.out
        assert "Pipeline Mode" in captured.out
        assert "Split Files Mode" in captured.out
