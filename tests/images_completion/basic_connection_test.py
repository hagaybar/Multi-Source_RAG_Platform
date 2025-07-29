from openai import OpenAI
import os

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o", messages=[{"role": "user", "content": "Say hello"}]
)

print(response.choices[0].message.content)
