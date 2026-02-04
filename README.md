# media-utils

An AI agent for image generation using natural language. Powered by LangGraph, Qwen LLM, and Z-Image-Turbo.

## Features

- **Natural Language Interface**: Just describe what you want - "Generate a sunset over mountains"
- **LangGraph Agent**: LLM-powered decision making for intelligent image generation
- **Z-Image-Turbo**: High-quality 6B parameter diffusion model
- **Memory Optimized**: CPU offloading for efficient GPU usage
- **Simple API**: One class, easy to use

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
python -m media_utils.utils.downloader all
```

### 4. Run the agent

```bash
python examples/example.py
```

## Usage

### Natural Language (via LLM)

```python
from media_utils import MediaAgent

agent = MediaAgent()

# The LLM interprets your request and generates appropriate images
result = agent.run("Generate a sunset over mountains")
print(result)  # "Image saved to: output/generated_xxx.png"

result = agent.run("Create a cyberpunk city at night with neon lights")
print(result)

# Cleanup
agent.unload()
```

### Direct Generation (bypass LLM)

```python
from media_utils import MediaAgent

agent = MediaAgent()

# Generate directly without LLM interpretation
path = agent.generate(
    prompt="A serene Japanese garden with cherry blossoms",
    negative_prompt="blurry, low quality",
    resolution="1344x768",  # 16:9 landscape
    seed=42,
)
print(f"Saved to: {path}")

agent.unload()
```

### Context Manager (auto cleanup)

```python
from media_utils import MediaAgent

with MediaAgent() as agent:
    agent.run("Generate a forest landscape")
    agent.run("Create an ocean sunset")
# Models automatically unloaded
```

## Configuration

```python
agent = MediaAgent(
    output_dir="output",           # Where to save images
    llm_model="Qwen/Qwen2.5-7B-Instruct",  # LLM for reasoning
    image_mode="pipeline",         # "pipeline", "split", or "local"
    offload_mode="model",          # "none", "model", or "sequential"
    device="cuda",                 # "cuda" or "cpu"
)
```

### Memory Modes

| Mode | VRAM | Speed | Description |
|------|------|-------|-------------|
| `"none"` | ~16GB | Fastest | Full model on GPU |
| `"model"` | ~8-10GB | Good | CPU offload when idle (default) |
| `"sequential"` | ~4-6GB | Slowest | Maximum memory savings |

### Resolution Presets

| Resolution | Size | Aspect Ratio |
|------------|------|--------------|
| `"1024x1024"` | 1024×1024 | 1:1 (Square) |
| `"1344x768"` | 1344×768 | 16:9 (Landscape) |
| `"768x1344"` | 768×1344 | 9:16 (Portrait) |
| `"1152x896"` | 1152×896 | 4:3 (Landscape) |
| `"896x1152"` | 896×1152 | 3:4 (Portrait) |

## How It Works

```
User: "Create a sunset over mountains"
           │
           ▼
    ┌─────────────┐
    │  Qwen LLM   │  Interprets request, creates detailed prompt
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │  Z-Image    │  Generates high-quality image
    │   Turbo     │
    └──────┬──────┘
           │
           ▼
    output/generated_xxx.png
```

The agent uses Qwen to:
1. Understand your natural language request
2. Create a detailed, optimized prompt for image generation
3. Choose appropriate settings (resolution, negative prompts)
4. Execute the generation

## Requirements

- Python >= 3.10
- CUDA-capable GPU (16GB+ VRAM recommended, 8GB with offloading)
- PyTorch >= 2.0
- uv (for fast package installation)[^1]

[^1]: To install uv, see: https://docs.astral.sh/uv/getting-started/installation/

## License

MIT
