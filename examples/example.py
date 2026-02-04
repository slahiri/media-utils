#!/usr/bin/env python3
"""Example usage of MediaAgent."""

from media_utils import MediaAgent


def main():
    """Interactive mode - describe what you want to generate."""
    print("Media Agent - AI Image Generation")
    print("Describe what you want, type 'quit' to exit\n")

    with MediaAgent() as agent:
        while True:
            query = input("You: ").strip()
            if not query:
                continue
            if query.lower() in ["quit", "exit", "q"]:
                break
            print(f"Agent: {agent.run(query)}\n")


if __name__ == "__main__":
    main()
