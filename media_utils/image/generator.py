import os
from pathlib import Path
from typing import Any, Literal

import torch
from PIL import Image

from media_utils.config import load_config, get_torch_dtype


# Available schedulers for Z-Image
SCHEDULERS = {
    "flow_match_euler": "FlowMatchEulerDiscreteScheduler",
    "euler": "EulerDiscreteScheduler",
    "euler_ancestral": "EulerAncestralDiscreteScheduler",
    "dpmpp_2m": "DPMSolverMultistepScheduler",
    "dpmpp_2m_karras": "DPMSolverMultistepScheduler",
    "dpmpp_sde": "DPMSolverSDEScheduler",
    "dpmpp_sde_karras": "DPMSolverSDEScheduler",
    "ddim": "DDIMScheduler",
    "unipc": "UniPCMultistepScheduler",
}

# Common resolution presets (width x height)
RESOLUTIONS = {
    "1024x1024": (1024, 1024),   # Square (default)
    "1152x896": (1152, 896),     # Landscape 4:3
    "896x1152": (896, 1152),     # Portrait 3:4
    "1216x832": (1216, 832),     # Landscape 3:2
    "832x1216": (832, 1216),     # Portrait 2:3
    "1344x768": (1344, 768),     # Landscape 16:9
    "768x1344": (768, 1344),     # Portrait 9:16
    "1536x640": (1536, 640),     # Ultrawide 21:9
    "640x1536": (640, 1536),     # Tall 9:21
}


class ImageGenerator:
    """Image generator using Z-Image-Turbo model.

    Optimized for memory efficiency similar to ComfyUI:
    - Model CPU offload: Models move to GPU only when needed
    - Sequential CPU offload: Even more memory savings (slower)
    - VAE slicing/tiling: Reduces memory for VAE decoding
    - Configurable schedulers and sampling parameters
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        mode: Literal["pipeline", "local", "split"] = "pipeline",
        model_name: str | None = None,
        device: str | None = None,
        torch_dtype: str | None = None,
        offload_mode: Literal["none", "model", "sequential"] = "model",
        enable_vae_slicing: bool = True,
        enable_vae_tiling: bool = False,
        scheduler: str | None = None,
        use_karras_sigmas: bool = False,
        use_beta_sigmas: bool = False,
        keep_loaded: bool = True,
    ):
        """Initialize the image generator.

        Args:
            config_path: Path to config file. Uses default if not provided.
            mode: Loading mode:
                - "pipeline": Load from HuggingFace using diffusers pipeline
                - "local": Load from local models folder (split files)
                - "split": Load split files from HuggingFace cache
            model_name: Override model name from config (pipeline mode only).
            device: Override device from config.
            torch_dtype: Override torch dtype from config.
            offload_mode: Memory optimization mode (default: "model"):
                - "none": Load entire model to GPU (fastest, most VRAM)
                - "model": CPU offload - models move to GPU only when needed (recommended)
                - "sequential": Sequential CPU offload - maximum memory savings (slowest)
            enable_vae_slicing: Enable VAE slicing for lower memory (default: True).
            enable_vae_tiling: Enable VAE tiling for very large images (default: False).
            scheduler: Scheduler to use. Options:
                - "flow_match_euler" (default for Z-Image)
                - "euler", "euler_ancestral"
                - "dpmpp_2m", "dpmpp_2m_karras"
                - "dpmpp_sde", "dpmpp_sde_karras"
                - "ddim", "unipc"
            use_karras_sigmas: Use Karras noise schedule (better quality).
            use_beta_sigmas: Use Beta noise schedule (alternative).
            keep_loaded: Keep model loaded in memory after generation (default: True).
                - True: Model stays loaded for faster subsequent generations
                - False: Model unloads after each generation to free VRAM
        """
        self.config = load_config(config_path)
        self.config_path = config_path
        image_config = self.config["models"]["image"]

        self.mode = mode
        self.model_name = model_name or image_config["pipeline"]["name"]
        self.device = device or image_config["pipeline"]["device"]
        self.torch_dtype = get_torch_dtype(torch_dtype or image_config["pipeline"]["torch_dtype"])
        self.default_steps = image_config.get("default_steps", 8)
        self.default_size = image_config.get("default_size", [1024, 1024])
        self.guidance_scale = image_config.get("guidance_scale", 1.0)

        # Memory optimization settings
        self.offload_mode = offload_mode
        self.enable_vae_slicing = enable_vae_slicing
        self.enable_vae_tiling = enable_vae_tiling

        # Scheduler settings
        self.scheduler_name = scheduler
        self.use_karras_sigmas = use_karras_sigmas
        self.use_beta_sigmas = use_beta_sigmas

        # Model lifecycle
        self.keep_loaded = keep_loaded

        self._pipeline = None

    @property
    def pipeline(self):
        """Lazy load the pipeline."""
        if self._pipeline is None:
            self._load_pipeline()
        return self._pipeline

    def _get_project_root(self) -> Path:
        """Get the project root directory."""
        if self.config_path:
            return Path(self.config_path).parent
        # Default to the package's parent directory
        return Path(__file__).parent.parent.parent

    def _get_local_model_paths(self) -> dict[str, Path]:
        """Get local model file paths."""
        local_config = self.config["models"]["image"]["local"]
        project_root = self._get_project_root()

        return {
            "text_encoder": project_root / local_config["text_encoder"],
            "diffusion_model": project_root / local_config["diffusion_model"],
            "vae": project_root / local_config["vae"],
        }

    def _check_local_models_exist(self) -> bool:
        """Check if all local model files exist."""
        paths = self._get_local_model_paths()
        return all(p.exists() for p in paths.values())

    def _get_scheduler(self, scheduler_name: str):
        """Get a scheduler instance by name."""
        from diffusers import (
            FlowMatchEulerDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
            DPMSolverSDEScheduler,
            DDIMScheduler,
            UniPCMultistepScheduler,
        )

        scheduler_map = {
            "flow_match_euler": FlowMatchEulerDiscreteScheduler,
            "euler": EulerDiscreteScheduler,
            "euler_ancestral": EulerAncestralDiscreteScheduler,
            "dpmpp_2m": DPMSolverMultistepScheduler,
            "dpmpp_2m_karras": DPMSolverMultistepScheduler,
            "dpmpp_sde": DPMSolverSDEScheduler,
            "dpmpp_sde_karras": DPMSolverSDEScheduler,
            "ddim": DDIMScheduler,
            "unipc": UniPCMultistepScheduler,
        }

        if scheduler_name not in scheduler_map:
            available = ", ".join(scheduler_map.keys())
            raise ValueError(f"Unknown scheduler: {scheduler_name}. Available: {available}")

        scheduler_class = scheduler_map[scheduler_name]

        # Get config from current scheduler
        scheduler_config = self._pipeline.scheduler.config

        # Build kwargs based on scheduler type
        kwargs = {}

        # Karras sigmas (for compatible schedulers)
        if self.use_karras_sigmas or "karras" in scheduler_name:
            if hasattr(scheduler_class, "use_karras_sigmas") or scheduler_name in ["dpmpp_2m_karras", "dpmpp_sde_karras"]:
                kwargs["use_karras_sigmas"] = True

        # Beta sigmas (for flow match scheduler)
        if self.use_beta_sigmas and scheduler_name == "flow_match_euler":
            kwargs["use_beta_sigmas"] = True

        return scheduler_class.from_config(scheduler_config, **kwargs)

    def _load_pipeline(self):
        """Load the Z-Image pipeline with memory optimizations."""
        from diffusers import ZImagePipeline

        cache_dir = self.config.get("paths", {}).get("cache_dir")
        if cache_dir:
            cache_dir = os.path.expanduser(cache_dir)

        if self.mode == "pipeline":
            # Standard diffusers pipeline loading
            self._pipeline = ZImagePipeline.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                cache_dir=cache_dir,
            )
        elif self.mode == "local":
            # Load from local split files
            if not self._check_local_models_exist():
                missing = [str(p) for p in self._get_local_model_paths().values() if not p.exists()]
                raise FileNotFoundError(
                    f"Local model files not found: {missing}. "
                    "Run download_image_model(mode='split', copy_to_local=True) first."
                )
            paths = self._get_local_model_paths()
            self._pipeline = self._load_from_split_files(paths)
        elif self.mode == "split":
            # Load from HuggingFace cache (split files)
            from media_utils.utils.downloader import get_split_file_paths
            paths = get_split_file_paths(self.config_path)
            self._pipeline = self._load_from_split_files(paths)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Apply scheduler if specified
        if self.scheduler_name:
            self._pipeline.scheduler = self._get_scheduler(self.scheduler_name)
            print(f"Using scheduler: {self.scheduler_name}")

        # Apply memory optimizations
        self._apply_memory_optimizations()

    def _apply_memory_optimizations(self):
        """Apply memory optimization settings to the pipeline."""
        # Apply offload mode
        if self.offload_mode == "model":
            # Model CPU offload - models move to GPU only when needed
            # This is similar to how ComfyUI handles memory
            self._pipeline.enable_model_cpu_offload()
            print("Enabled model CPU offload (recommended for shared GPU usage)")
        elif self.offload_mode == "sequential":
            # Sequential CPU offload - maximum memory savings
            self._pipeline.enable_sequential_cpu_offload()
            print("Enabled sequential CPU offload (maximum memory savings)")
        else:
            # No offload - load everything to GPU
            self._pipeline.to(self.device)
            print(f"Loaded full model to {self.device} (no CPU offload)")

        # VAE memory optimizations
        if self.enable_vae_slicing:
            self._pipeline.enable_vae_slicing()
            print("Enabled VAE slicing")

        if self.enable_vae_tiling:
            self._pipeline.enable_vae_tiling()
            print("Enabled VAE tiling")

    def _load_from_split_files(self, paths: dict[str, Path]) -> Any:
        """Load pipeline from split safetensor files."""
        from diffusers import ZImagePipeline

        # Fallback to pipeline mode with a warning
        print("Note: Split file loading requires specific model architecture setup.")
        print("Falling back to pipeline mode for now.")

        cache_dir = self.config.get("paths", {}).get("cache_dir")
        if cache_dir:
            cache_dir = os.path.expanduser(cache_dir)

        return ZImagePipeline.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
            cache_dir=cache_dir,
        )

    def _get_generator_device(self) -> str:
        """Get the correct device for the random generator.

        When using CPU offload, the generator should be on CPU.
        """
        if self.offload_mode in ("model", "sequential"):
            return "cpu"
        return self.device

    def set_scheduler(self, scheduler_name: str):
        """Change the scheduler on the fly.

        Args:
            scheduler_name: Name of the scheduler. Options:
                - "flow_match_euler" (default for Z-Image)
                - "euler", "euler_ancestral"
                - "dpmpp_2m", "dpmpp_2m_karras"
                - "dpmpp_sde", "dpmpp_sde_karras"
                - "ddim", "unipc"
        """
        if self._pipeline is None:
            self.scheduler_name = scheduler_name
        else:
            self._pipeline.scheduler = self._get_scheduler(scheduler_name)
            print(f"Switched to scheduler: {scheduler_name}")

    def generate(
        self,
        prompt: str,
        negative_prompt: str | None = None,
        resolution: str | tuple[int, int] | None = None,
        width: int | None = None,
        height: int | None = None,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        seed: int | None = None,
        **kwargs: Any,
    ) -> Image.Image:
        """Generate an image from a text prompt.

        Args:
            prompt: Text description of the image to generate.
            negative_prompt: What to avoid in the image (optional).
            resolution: Image resolution as "WxH" string, preset name, or (width, height) tuple.
                Presets: "1024x1024", "1152x896", "896x1152", "1216x832", "832x1216",
                         "1344x768", "768x1344", "1536x640", "640x1536"
            width: Image width in pixels (overrides resolution).
            height: Image height in pixels (overrides resolution).
            num_inference_steps: Number of denoising steps. Defaults to config value.
            guidance_scale: CFG scale. Defaults to config value (1.0 for Turbo).
            seed: Random seed for reproducibility.
            **kwargs: Additional arguments passed to the pipeline.

        Returns:
            Generated PIL Image.
        """
        # Parse resolution
        if resolution is not None:
            if isinstance(resolution, str):
                if resolution in RESOLUTIONS:
                    res_width, res_height = RESOLUTIONS[resolution]
                elif "x" in resolution:
                    # Parse "WxH" format
                    parts = resolution.lower().split("x")
                    res_width, res_height = int(parts[0]), int(parts[1])
                else:
                    raise ValueError(f"Invalid resolution: {resolution}. Use 'WxH' format or preset: {list(RESOLUTIONS.keys())}")
            else:
                # Tuple (width, height)
                res_width, res_height = resolution
            width = width or res_width
            height = height or res_height

        # Use defaults if not specified
        width = width or self.default_size[0]
        height = height or self.default_size[1]
        num_inference_steps = num_inference_steps or self.default_steps
        guidance_scale = guidance_scale if guidance_scale is not None else self.guidance_scale

        generator = None
        if seed is not None:
            # Use correct device for generator based on offload mode
            gen_device = self._get_generator_device()
            generator = torch.Generator(gen_device).manual_seed(seed)

        # Build pipeline kwargs
        pipe_kwargs = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
            **kwargs,
        }

        # Add negative prompt if provided
        if negative_prompt:
            pipe_kwargs["negative_prompt"] = negative_prompt

        result = self.pipeline(**pipe_kwargs)

        image = result.images[0]

        # Unload model if keep_loaded is False
        if not self.keep_loaded:
            self.unload()

        return image

    def generate_batch(
        self,
        prompts: list[str],
        **kwargs: Any,
    ) -> list[Image.Image]:
        """Generate multiple images from text prompts.

        Args:
            prompts: List of text descriptions.
            **kwargs: Arguments passed to generate().

        Returns:
            List of generated PIL Images.
        """
        # Temporarily keep loaded during batch, then respect keep_loaded setting
        original_keep_loaded = self.keep_loaded
        self.keep_loaded = True  # Keep loaded during batch

        images = [self.generate(prompt, **kwargs) for prompt in prompts]

        # Restore and apply original setting
        self.keep_loaded = original_keep_loaded
        if not self.keep_loaded:
            self.unload()

        return images

    def unload(self):
        """Unload the model from memory."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("Model unloaded from memory")

    @staticmethod
    def list_schedulers() -> list[str]:
        """List available schedulers."""
        return list(SCHEDULERS.keys())

    @staticmethod
    def list_resolutions() -> dict[str, tuple[int, int]]:
        """List available resolution presets."""
        return RESOLUTIONS.copy()

    def __enter__(self):
        """Context manager entry - returns self."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - unloads model."""
        self.unload()
        return False
