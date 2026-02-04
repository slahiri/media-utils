"""Example usage of image-gen library."""

from media_utils import ImageGenerator, QwenLLM, download_models, list_available_models


def example_image_generation_pipeline():
    """Example: Generate an image using diffusers pipeline."""
    print("=== Image Generation (Pipeline Mode) ===")

    # Initialize the generator (downloads model on first use)
    gen = ImageGenerator(mode="pipeline")

    # Generate an image
    image = gen.generate(
        prompt="A serene mountain landscape at sunset with a crystal clear lake",
        seed=42,
    )

    # Save the image
    image.save("output/generated_pipeline.png")
    print("Image saved to: output/generated_pipeline.png")


def example_image_generation_local():
    """Example: Generate an image using local model files."""
    print("=== Image Generation (Local Mode) ===")

    # Initialize the generator using local model files
    # Requires models to be downloaded first with copy_to_local=True
    gen = ImageGenerator(mode="local")

    # Generate an image
    image = gen.generate(
        prompt="A cyberpunk city at night with neon lights",
        seed=123,
    )

    # Save the image
    image.save("output/generated_local.png")
    print("Image saved to: output/generated_local.png")


def example_llm_generate():
    """Example: Generate text with Qwen."""
    print("\n=== LLM Text Generation Example ===")

    # Initialize the LLM
    llm = QwenLLM()

    # Generate text
    response = llm.generate(
        prompt="Explain quantum computing in simple terms:",
        max_new_tokens=200,
    )

    print(f"Response: {response}")


def example_llm_chat():
    """Example: Chat with Qwen."""
    print("\n=== LLM Chat Example ===")

    # Initialize the LLM
    llm = QwenLLM()

    # Chat conversation
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What are the main benefits of renewable energy?"},
    ]

    response = llm.chat(messages)
    print(f"Assistant: {response}")


def example_download_pipeline():
    """Download models using pipeline mode."""
    print("=== Downloading Models (Pipeline) ===")
    download_models(image_mode="pipeline")


def example_download_split():
    """Download models using split files mode and copy to local."""
    print("=== Downloading Models (Split + Local) ===")
    download_models(image_mode="split", copy_to_local=True)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        example = sys.argv[1]
        examples = {
            "image": example_image_generation_pipeline,
            "image-local": example_image_generation_local,
            "generate": example_llm_generate,
            "chat": example_llm_chat,
            "download": example_download_pipeline,
            "download-split": example_download_split,
            "list": list_available_models,
        }

        if example in examples:
            examples[example]()
        else:
            print(f"Unknown example: {example}")
            print(f"Available: {', '.join(examples.keys())}")
    else:
        print("Usage: python usage.py <example>")
        print()
        print("Examples:")
        print("  list           - List available models")
        print("  download       - Download models (pipeline mode)")
        print("  download-split - Download models (split files, copy to local)")
        print("  image          - Generate an image (pipeline mode)")
        print("  image-local    - Generate an image (local models)")
        print("  generate       - Generate text with LLM")
        print("  chat           - Chat with the LLM")
