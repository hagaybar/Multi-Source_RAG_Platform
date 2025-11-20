import os
from openai import OpenAI

# Try to get API key from environment variables
# Windows: OPEN_AI, WSL/Linux: OPENAI_API_KEY (standard)
api_key = os.getenv("OPEN_AI") or os.getenv("OPENAI_API_KEY")

if api_key:
    client = OpenAI(api_key=api_key)
else:
    # Create client without key - will fail gracefully when actually used
    # This allows imports to succeed even without API key
    client = None


def translate_to_english(text: str) -> str:
    """
    Translates the input text to English using OpenAI's ChatCompletion API
    (v1.x syntax).
    Uses a low-cost model (gpt-3.5-turbo) and deterministic settings.

    Requires OPEN_AI (Windows) or OPENAI_API_KEY (Linux/WSL) environment variable.

    Returns the translated string, or the original text if translation fails.
    """
    if not text or not isinstance(text, str):
        return text

    # Check if API key is available
    if client is None:
        print(
            "[WARN] OpenAI API key not found. Please set either:\n"
            "  - OPEN_AI environment variable (Windows)\n"
            "  - OPENAI_API_KEY environment variable (Linux/WSL/Standard)\n"
            "Get your API key from: https://platform.openai.com/api-keys\n"
            "Returning original text without translation."
        )
        return text

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Translate the user's input to English. "
                        "Do not explain. "
                        "Keep technical and domain-specific terms unchanged."
                    ),
                },
                {"role": "user", "content": text},
            ],
            temperature=0,
            max_tokens=1000,
        )
        translated = response.choices[0].message.content.strip()
        return translated

    except Exception as e:
        print(f"[WARN] OpenAI translation failed: {e}")
        return text  # Fallback to original if error occurs
