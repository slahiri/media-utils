"""Test Qwen LLM generation."""

from media_utils import QwenLLM

print("Initializing QwenLLM...")
llm = QwenLLM()

# Test text generation
print("\n=== Text Generation ===")
response = llm.generate(
    prompt="Explain quantum computing in simple terms:",
    max_new_tokens=200,
)
print(f"Response: {response}")

# Test chat
print("\n=== Chat ===")
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
]
response = llm.chat(messages)
print(f"Assistant: {response}")
