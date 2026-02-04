# media-utils

A Python library for media utilities including image generation using Z-Image-Turbo and LLM text generation using Qwen.

## Features

- **Image Generation**: Z-Image-Turbo (6B params) - photorealistic images with bilingual text rendering
- **LLM**: Qwen2.5-7B-Instruct for text generation and chat
- **Memory Optimized**: ComfyUI-style CPU offloading for efficient GPU usage
- **Configurable Schedulers**: Support for multiple samplers and noise schedules
- **Flexible Loading**: HuggingFace pipeline, split files (ComfyUI-style), or local models
- **Config-driven**: YAML configuration for models, paths, and generation defaults

## Setup

### 1. Create and activate virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows
```

### 2. Install dependencies

```bash
uv pip install -e .
```

### 3. Download models

```bash
# Download all models (pipeline mode)
python -m media_utils.utils.downloader all

# Or download split files and copy to local models/ folder
python -m media_utils.utils.downloader all split --local
```

### 4. Run an example

```bash
python examples/test_pipeline.py
```

### 5. Check the output

Generated images are saved to the `output/` folder.

## Quick Start

### Generate Images

```python
from media_utils import ImageGenerator

# Basic usage (with CPU offload for better GPU sharing)
gen = ImageGenerator(mode="pipeline")

# Generate an image
image = gen.generate(
    prompt="A serene mountain landscape at sunset",
    seed=42,
)
image.save("output/image.png")

# Free GPU memory when done
gen.unload()
```

### Advanced Options

```python
gen = ImageGenerator(
    mode="pipeline",
    # Memory optimization
    offload_mode="model",       # "none", "model" (recommended), "sequential"
    enable_vae_slicing=True,    # Reduce VAE memory
    keep_loaded=True,           # Keep model loaded for multiple generations
    # Scheduler options
    scheduler="dpmpp_sde_karras",  # Better quality
    use_karras_sigmas=True,     # Better noise schedule
)

image = gen.generate(
    prompt="A cyberpunk city at night",
    negative_prompt="blurry, low quality",
    resolution="1344x768",      # 16:9 landscape
    num_inference_steps=8,      # 8-9 for Turbo
    guidance_scale=1.0,         # 1.0 for Turbo models
    seed=123,
)
```

### Context Manager (Auto Cleanup)

```python
from media_utils import ImageGenerator

with ImageGenerator() as gen:
    image = gen.generate("A mountain landscape")
    image.save("output/image.png")
# Model automatically unloaded
```

### Use LLM

```python
from media_utils import QwenLLM

llm = QwenLLM()

# Text generation
response = llm.generate("Explain quantum computing:")

# Chat format
response = llm.chat([
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is AI?"},
])
```

## Memory Optimization

Similar to ComfyUI, this library supports CPU offloading for efficient GPU usage:

| Mode | VRAM Usage | Speed | Description |
|------|------------|-------|-------------|
| `"none"` | ~16GB | Fastest | Full model on GPU |
| `"model"` | ~8-10GB | Good | Models move to GPU only when needed (default) |
| `"sequential"` | ~4-6GB | Slowest | Maximum memory savings |

## Model Lifecycle

| `keep_loaded` | Behavior |
|---------------|----------|
| `True` (default) | Model stays loaded, call `unload()` manually |
| `False` | Model unloads after each `generate()` call |

## Schedulers

Available schedulers for different quality/speed tradeoffs:

| Scheduler | Best For | Steps |
|-----------|----------|-------|
| `flow_match_euler` | Default (Z-Image native) | 8 |
| `euler` | Fast generation | 8 |
| `euler_ancestral` | More diversity | 8 |
| `dpmpp_sde_karras` | Better quality | 8-10 |
| `dpmpp_2m_karras` | Balanced | 10-15 |
| `ddim` | Deterministic | 20+ |
| `unipc` | Fast convergence | 5-10 |

```python
# List all schedulers
print(ImageGenerator.list_schedulers())

# Change scheduler on the fly
gen.set_scheduler("dpmpp_sde_karras")
```

## Generation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `resolution` | "1024x1024" | Image size as preset, "WxH" string, or (w,h) tuple |
| `num_inference_steps` | 8 | Denoising steps (8-9 for Turbo) |
| `guidance_scale` | 1.0 | CFG scale (1.0 for Turbo) |
| `negative_prompt` | None | What to avoid in the image |
| `seed` | None | Random seed for reproducibility |

**Resolution presets:**
| Preset | Size | Aspect Ratio |
|--------|------|--------------|
| `"1024x1024"` | 1024×1024 | 1:1 (Square) |
| `"1152x896"` | 1152×896 | 4:3 (Landscape) |
| `"896x1152"` | 896×1152 | 3:4 (Portrait) |
| `"1344x768"` | 1344×768 | 16:9 (Landscape) |
| `"768x1344"` | 768×1344 | 9:16 (Portrait) |

```python
# Resolution options
image = gen.generate(prompt, resolution="1344x768")       # Preset
image = gen.generate(prompt, resolution="800x600")        # Custom "WxH"
image = gen.generate(prompt, resolution=(1280, 720))      # Tuple
image = gen.generate(prompt, width=1024, height=768)      # Individual params
```

## Models

### Z-Image-Turbo (Image Generation)

| Component | File | Size |
|-----------|------|------|
| Text Encoder | `qwen_3_4b.safetensors` | ~8GB |
| Diffusion Model | `z_image_turbo_bf16.safetensors` | ~12GB |
| VAE | `ae.safetensors` | ~335MB |

**Sources:**
- Pipeline: `Tongyi-MAI/Z-Image-Turbo`
- Split files: `Comfy-Org/z_image_turbo`

### Qwen LLM

- Default: `Qwen/Qwen2.5-7B-Instruct`
- Alternatives: `Qwen2.5-3B-Instruct` (smaller), `Qwen2.5-14B-Instruct` (larger)

## Project Structure

```
media-utils/
├── config.yaml              # Model configurations
├── pyproject.toml           # Package dependencies
├── output/                  # Generated images
├── models/                  # Local models folder
│   ├── text_encoders/
│   ├── diffusion_models/
│   ├── vae/
│   └── llm/
├── media_utils/
│   ├── config.py            # Config loader
│   ├── image/
│   │   └── generator.py     # ImageGenerator class
│   ├── llm/
│   │   └── qwen.py          # QwenLLM class
│   └── utils/
│       └── downloader.py    # Model download utilities
├── examples/
│   ├── test_pipeline.py     # Test pipeline mode
│   ├── test_split.py        # Test split files mode
│   ├── test_qwen.py         # Test LLM
│   └── usage.py             # Full examples
└── tests/                   # Unit tests
```

## CLI Commands

```bash
# List available models
python -m media_utils.utils.downloader list

# Download all models
python -m media_utils.utils.downloader all [pipeline|split] [--local]

# Download image model only
python -m media_utils.utils.downloader image [pipeline|split] [--local]

# Download LLM only
python -m media_utils.utils.downloader llm [--local]
```

## Requirements

- Python >= 3.10
- CUDA-capable GPU (16GB+ VRAM recommended, 8GB with offloading)
- PyTorch >= 2.0
- uv (for fast package installation)[^1]

[^1]: To install uv, see: https://docs.astral.sh/uv/getting-started/installation/

## License

MIT
