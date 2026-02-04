import os
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download, hf_hub_download

from media_utils.config import load_config


def get_project_root(config_path: str | Path | None = None) -> Path:
    """Get the project root directory."""
    if config_path:
        return Path(config_path).parent
    # Default: find config.yaml location
    current = Path(__file__).parent
    while current != current.parent:
        if (current / "config.yaml").exists():
            return current
        current = current.parent
    return Path(__file__).parent.parent.parent


def get_split_file_paths(config_path: str | Path | None = None) -> dict[str, Path]:
    """Get paths to split files in HuggingFace cache.

    Args:
        config_path: Path to config file.

    Returns:
        Dictionary mapping component names to cached file paths.
    """
    config = load_config(config_path)
    split_config = config["models"]["image"]["split_files"]
    repo_id = split_config["repo_id"]
    models_dir = config.get("paths", {}).get("cache_dir")

    if models_dir:
        models_dir = os.path.expanduser(models_dir)

    paths = {}
    for component in ["text_encoder", "diffusion_model", "vae"]:
        file_path = split_config[component]
        # Get the cached path
        cached_path = hf_hub_download(
            repo_id=repo_id,
            filename=file_path,
            cache_dir=models_dir,
            local_files_only=True,
        )
        paths[component] = Path(cached_path)

    return paths


def download_image_model_pipeline(
    config_path: str | Path | None = None,
    model_name: str | None = None,
    **kwargs,  # Accept extra kwargs for compatibility
) -> str:
    """Download the Z-Image model using diffusers pipeline approach.

    Args:
        config_path: Path to config file.
        model_name: Override model name from config.

    Returns:
        Path to downloaded model.
    """
    config = load_config(config_path)
    model_name = model_name or config["models"]["image"]["pipeline"]["name"]
    cache_dir = config.get("paths", {}).get("cache_dir")

    if cache_dir:
        cache_dir = os.path.expanduser(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)

    print(f"Downloading image model (pipeline): {model_name}")

    path = snapshot_download(
        repo_id=model_name,
        cache_dir=cache_dir,
    )

    print(f"Image model downloaded to: {path}")
    return path


def download_image_model_split(
    config_path: str | Path | None = None,
    copy_to_local: bool = False,
    include_quantized: bool = False,
) -> dict[str, str]:
    """Download Z-Image split model files (ComfyUI style).

    Args:
        config_path: Path to config file.
        copy_to_local: Copy downloaded files to local models/ folder.
        include_quantized: Also download FP8/FP4 quantized versions.

    Returns:
        Dictionary mapping component names to file paths.
    """
    config = load_config(config_path)
    split_config = config["models"]["image"]["split_files"]
    local_config = config["models"]["image"]["local"]
    repo_id = split_config["repo_id"]

    cache_dir = config.get("paths", {}).get("cache_dir")
    if cache_dir:
        cache_dir = os.path.expanduser(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)

    files_to_download = [
        ("text_encoder", split_config["text_encoder"], local_config["text_encoder"]),
        ("diffusion_model", split_config["diffusion_model"], local_config["diffusion_model"]),
        ("vae", split_config["vae"], local_config["vae"]),
    ]

    if include_quantized:
        if "text_encoder_fp8" in split_config:
            files_to_download.append((
                "text_encoder_fp8",
                split_config["text_encoder_fp8"],
                local_config["text_encoder"].replace(".safetensors", "_fp8.safetensors")
            ))
        if "diffusion_model_fp4" in split_config:
            files_to_download.append((
                "diffusion_model_fp4",
                split_config["diffusion_model_fp4"],
                local_config["diffusion_model"].replace(".safetensors", "_fp4.safetensors")
            ))

    paths = {}
    project_root = get_project_root(config_path)

    print(f"Downloading split model files from: {repo_id}")
    print("-" * 50)

    for component_name, remote_path, local_path in files_to_download:
        print(f"  Downloading {component_name}: {remote_path}")
        cached_path = hf_hub_download(
            repo_id=repo_id,
            filename=remote_path,
            cache_dir=cache_dir,
        )
        paths[component_name] = cached_path
        print(f"    Cached at: {cached_path}")

        if copy_to_local:
            local_full_path = project_root / local_path
            local_full_path.parent.mkdir(parents=True, exist_ok=True)

            if not local_full_path.exists():
                print(f"    Copying to: {local_full_path}")
                shutil.copy2(cached_path, local_full_path)
            else:
                print(f"    Already exists: {local_full_path}")

            paths[f"{component_name}_local"] = str(local_full_path)

    print("-" * 50)
    print("Split model files downloaded successfully!")

    if copy_to_local:
        print(f"\nModels copied to: {project_root / 'models'}")

    return paths


def download_image_model(
    config_path: str | Path | None = None,
    mode: str = "pipeline",
    **kwargs,
) -> str | dict[str, str]:
    """Download the Z-Image model.

    Args:
        config_path: Path to config file.
        mode: "pipeline" for diffusers approach, "split" for ComfyUI-style split files.
        **kwargs: Additional arguments passed to the specific download function.
            - copy_to_local (bool): For split mode, copy files to local models/ folder.
            - include_quantized (bool): For split mode, also download quantized versions.

    Returns:
        Path to downloaded model (pipeline) or dict of paths (split).
    """
    if mode == "pipeline":
        return download_image_model_pipeline(config_path, **kwargs)
    elif mode == "split":
        return download_image_model_split(config_path, **kwargs)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'pipeline' or 'split'.")


def download_llm_model(
    config_path: str | Path | None = None,
    model_name: str | None = None,
    copy_to_local: bool = False,
) -> str:
    """Download the Qwen LLM model.

    Args:
        config_path: Path to config file.
        model_name: Override model name from config.
        copy_to_local: Copy to local models/llm folder.

    Returns:
        Path to downloaded model.
    """
    config = load_config(config_path)
    model_name = model_name or config["models"]["llm"]["name"]
    cache_dir = config.get("paths", {}).get("cache_dir")

    if cache_dir:
        cache_dir = os.path.expanduser(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)

    print(f"Downloading LLM model: {model_name}")

    path = snapshot_download(
        repo_id=model_name,
        cache_dir=cache_dir,
    )

    print(f"LLM model downloaded to: {path}")

    if copy_to_local:
        project_root = get_project_root(config_path)
        local_path = project_root / config["models"]["llm"].get("local_path", f"models/llm/{model_name.split('/')[-1]}")
        local_path.parent.mkdir(parents=True, exist_ok=True)

        if not local_path.exists():
            print(f"Copying to: {local_path}")
            shutil.copytree(path, local_path)
        else:
            print(f"Already exists: {local_path}")

        return str(local_path)

    return path


def download_models(
    config_path: str | Path | None = None,
    image_mode: str = "pipeline",
    copy_to_local: bool = False,
) -> dict[str, str | dict]:
    """Download all configured models.

    Args:
        config_path: Path to config file.
        image_mode: "pipeline" or "split" for image model download approach.
        copy_to_local: Copy downloaded models to local models/ folder.

    Returns:
        Dictionary mapping model type to download path(s).
    """
    paths = {}

    print("=" * 50)
    print("Downloading all models...")
    print("=" * 50)
    print()

    paths["image"] = download_image_model(
        config_path,
        mode=image_mode,
        copy_to_local=copy_to_local,
    )
    print()

    paths["llm"] = download_llm_model(
        config_path,
        copy_to_local=copy_to_local,
    )
    print()

    print("=" * 50)
    print("All models downloaded successfully!")
    print("=" * 50)
    return paths


def list_available_models() -> None:
    """Print information about available models."""
    print("""
Z-Image-Turbo Models
====================

Pipeline Mode (Diffusers):
  - Tongyi-MAI/Z-Image-Turbo (full pipeline, ~16GB)

Split Files Mode (ComfyUI):
  From Comfy-Org/z_image_turbo:
  - qwen_3_4b.safetensors (text encoder, ~8GB)
  - z_image_turbo_bf16.safetensors (diffusion model, ~12GB)
  - ae.safetensors (VAE, ~335MB)

  Quantized variants:
  - qwen_3_4b_fp8_mixed.safetensors (FP8 text encoder)
  - z_image_turbo_nvfp4.safetensors (FP4 diffusion model)

LLM Models
==========
  - Qwen/Qwen2.5-7B-Instruct (default)
  - Qwen/Qwen2.5-3B-Instruct (smaller)
  - Qwen/Qwen2.5-14B-Instruct (larger)

Download Commands
=================
  # Download all (pipeline mode)
  python -m media_utils.utils.downloader all

  # Download all (split files, copy to local)
  python -m media_utils.utils.downloader all split --local

  # Download image model only (split files)
  python -m media_utils.utils.downloader image split

  # Download LLM only
  python -m media_utils.utils.downloader llm
""")


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]

    if not args or args[0] == "help":
        print("Usage: python -m media_utils.utils.downloader <command> [options]")
        print()
        print("Commands:")
        print("  list                    - List available models")
        print("  all [mode] [--local]    - Download all models")
        print("  image [mode] [--local]  - Download image model only")
        print("  llm [--local]           - Download LLM model only")
        print()
        print("Options:")
        print("  mode     - 'pipeline' (default) or 'split'")
        print("  --local  - Copy models to local models/ folder")
        sys.exit(0)

    cmd = args[0]
    mode = "pipeline"
    copy_local = "--local" in args

    if len(args) > 1 and args[1] in ["pipeline", "split"]:
        mode = args[1]

    if cmd == "list":
        list_available_models()
    elif cmd == "all":
        download_models(image_mode=mode, copy_to_local=copy_local)
    elif cmd == "image":
        download_image_model(mode=mode, copy_to_local=copy_local)
    elif cmd == "llm":
        download_llm_model(copy_to_local=copy_local)
    else:
        print(f"Unknown command: {cmd}")
        print("Run with 'help' for usage information.")
