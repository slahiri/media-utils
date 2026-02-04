"""Test image generation using split files mode."""

from media_utils import ImageGenerator

print("Initializing ImageGenerator (split mode)...")
gen = ImageGenerator(mode="split")

print("Generating image...")
image = gen.generate(
    prompt="A cyberpunk city at night with neon lights",
    seed=123,
)

image.save("output/test_split.png")
print("Image saved to: output/test_split.png")
