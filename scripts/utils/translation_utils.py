from openai import OpenAI

client = OpenAI()

def translate_to_english(text: str) -> str:
    """
    Translates the input text to English using OpenAI's ChatCompletion API (v1.x syntax).
    Uses a low-cost model (gpt-3.5-turbo) and deterministic settings.

    Returns the translated string, or the original text if translation fails.
    """
    if not text or not isinstance(text, str):
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
                {"role": "user", "content": text}
            ],
            temperature=0,
            max_tokens=1000
        )
        translated = response.choices[0].message.content.strip()
        return translated

    except Exception as e:
        print(f"[WARN] OpenAI translation failed: {e}")
        return text  # Fallback to original if error occurs
