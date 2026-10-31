"""Quick connectivity check for the OpenAI-compatible LLM endpoint.

Usage:
    export LLM_API_KEY=sk-xxx                       # your own key
    export LLM_API_BASE=https://api.openai.com/v1   # or any OpenAI-compatible endpoint
    python scripts/connect.py
"""
import os

from openai import OpenAI

client = OpenAI(
    base_url=os.getenv("LLM_API_BASE", "https://api.openai.com/v1"),
    # replace sk-xxx with your own key (or set LLM_API_KEY)
    api_key=os.getenv("LLM_API_KEY", "sk-xxx"),
)
completion = client.chat.completions.create(
    model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ],
)
print(completion.choices[0].message)
