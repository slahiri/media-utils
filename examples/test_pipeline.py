"""Test image generation using pipeline mode with memory optimizations."""

from media_utils import ImageGenerator

print("Available schedulers:", ImageGenerator.list_schedulers())
print("Available resolutions:", ImageGenerator.list_resolutions())

print("\nInitializing ImageGenerator (pipeline mode with CPU offload)...")
gen = ImageGenerator(
    mode="pipeline",
    offload_mode="model",       # "none", "model", or "sequential"
    enable_vae_slicing=True,
    # Scheduler options:
    # scheduler="flow_match_euler",  # default for Z-Image
    # scheduler="dpmpp_sde_karras",  # better quality
    # use_karras_sigmas=True,        # better noise schedule
    # Model lifecycle:
    keep_loaded=True,           # True: keep model loaded for faster subsequent generations
                                # False: unload after each generation to free VRAM
)

print("\nGenerating image...")
image = gen.generate(
    prompt="A serene mountain landscape at sunset with a crystal clear lake",
    # negative_prompt="blurry, low quality",  # optional
    # Resolution options (pick one):
    resolution="1024x1024",       # preset or "WxH" string
    # resolution=(1024, 1024),    # tuple (width, height)
    # width=1024, height=1024,    # individual params
    num_inference_steps=8,        # 8-9 recommended for Turbo
    guidance_scale=1.0,           # 1.0 for Turbo models
    seed=42,
)

image.save("output/test_pipeline.png")
print(f"Image saved to: output/test_pipeline.png ({image.size[0]}x{image.size[1]})")

# Manual unload (only needed if keep_loaded=True and you want to free VRAM)
gen.unload()
