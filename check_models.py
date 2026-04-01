# Create check_models.py and run it

import os
from dotenv import load_dotenv
load_dotenv()

import anthropic

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Test each model
models_to_test = [
    "claude-3-5-haiku-20241022",
    "claude-3-5-sonnet-20241022",
    "claude-haiku-4-5",
    "claude-sonnet-4-5",
]

print("Testing models...\n")
for model in models_to_test:
    try:
        message = client.messages.create(
            model=model,
            max_tokens=10,
            messages=[{"role": "user", "content": "Hi"}]
        )
        print(f"✅ {model} works")
    except Exception as e:
        print(f"❌ {model} failed: {e}")