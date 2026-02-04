#!/usr/bin/env python3
"""Example script for the MediaAgent."""

from media_utils import MediaAgent


def main():
    """Run the media agent interactively."""
    print("=" * 50)
    print("Media Agent - Image Generation & OCR")
    print("=" * 50)
    print()
    print("Commands:")
    print("  - Generate images: 'Create an image of a sunset'")
    print("  - OCR: 'Extract text from document.png'")
    print("  - Type 'quit' to exit")
    print()

    # Initialize the agent
    agent = MediaAgent(
        output_dir="output",
        ocr_quantization="4bit",
    )

    try:
        while True:
            query = input("\nYou: ").strip()

            if not query:
                continue

            if query.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            print("\nAgent: ", end="", flush=True)
            response = agent.run(query)
            print(response)

    finally:
        # Cleanup
        agent.unload()
        print("\nModels unloaded.")


def example_generate_image():
    """Example: Generate an image."""
    print("Generating image...")

    with MediaAgent() as agent:
        result = agent.run("Generate a photorealistic image of a mountain landscape at sunset with a lake reflection")
        print(result)


def example_ocr():
    """Example: Extract text from an image."""
    print("Extracting text...")

    with MediaAgent() as agent:
        result = agent.run("Extract the text from output/test_document.png using markdown mode")
        print(result)


def example_workflow():
    """Example: Multi-step workflow."""
    print("Running multi-step workflow...")

    with MediaAgent() as agent:
        # Generate an image
        print("\n1. Generating image...")
        result1 = agent.run("Create an image of a handwritten note that says 'Hello World'")
        print(result1)

        # Note: In a real scenario, you would then use OCR on the generated image
        # This is just a demonstration of the agent's capabilities


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "generate":
            example_generate_image()
        elif cmd == "ocr":
            example_ocr()
        elif cmd == "workflow":
            example_workflow()
        else:
            print(f"Unknown command: {cmd}")
            print("Usage: python test_agent.py [generate|ocr|workflow]")
    else:
        main()
