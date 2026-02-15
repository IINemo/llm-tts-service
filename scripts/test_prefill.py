import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

prompt = "Explain what a transformer model is in simple terms."
prefill = "A transformer model is a type of neural network that"

response = client.chat.completions.create(
    model="openai/gpt-4o-mini",
    extra_headers={
        "HTTP-Referer": "https://example.com",   # optional but recommended
        "X-Title": "PrefillTest"
    },
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": prefill},  # assistant prefill
    ],
    temperature=0.3,
)

output = response.choices[0].message.content

print("=== Prefill Output ===")
print(output)
print()

print("Starts with prefill:", output.strip().startswith(prefill))
print("Contains prefill:", prefill in output)
