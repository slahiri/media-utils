# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-02-04

### Added

- Initial release
- **Image Generation Module** (`media_utils.image.generator`)
  - `ImageGenerator` class with Z-Image-Turbo support
  - Three loading modes: `pipeline`, `local`, `split`
  - Configurable generation parameters (steps, size, guidance scale)
  - Batch generation support

- **LLM Module** (`media_utils.llm.qwen`)
  - `QwenLLM` class with Qwen2.5-7B-Instruct support
  - Text generation and chat interfaces
  - Local model path support

- **Model Downloader** (`media_utils.utils.downloader`)
  - Download from HuggingFace (pipeline or split files)
  - Copy to local `models/` folder
  - Support for quantized model variants (FP8, FP4)

- **Configuration**
  - YAML-based configuration (`config.yaml`)
  - Separate configs for pipeline and split file modes
  - Local model path configuration

- **Project Structure**
  - Local `models/` directory for storing downloaded models
  - Example usage scripts
  - Comprehensive `.gitignore` for model files
