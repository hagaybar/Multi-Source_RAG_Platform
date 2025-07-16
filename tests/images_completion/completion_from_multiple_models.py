import os
import time
import json
import base64
import mimetypes
from collections import defaultdict
from openai import OpenAI, OpenAIError

# --- CONFIGURATION ---
IMAGE_FOLDER = "./test_images"
OUTPUT_LOG = "./image_descriptions.jsonl"
MAX_RETRIES = 3
RETRY_DELAY_BASE = 5  # seconds

# Models to compare with token pricing
MODELS = [
    {"name": "gpt-4o", "price_input": 0.005, "price_output": 0.015},
    {"name": "gpt-4o-mini", "price_input": 0.0015, "price_output": 0.006},
    {"name": "gpt-4-turbo", "price_input": 0.01, "price_output": 0.03},
]

ALLOWED_MIME_TYPES = {"image/png", "image/jpeg", "image/webp", "image/gif"}

client = OpenAI()
cost_tracker = defaultdict(float)


def encode_image_as_data_url(image_path: str) -> tuple[str, str]:
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type not in ALLOWED_MIME_TYPES:
        raise ValueError(f"Unsupported image type: {mime_type} for file {image_path}")

    with open(image_path, "rb") as f:
        image_bytes = f.read()
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")
    return mime_type, f"data:{mime_type};base64,{encoded_image}"


def describe_image(image_path: str, model: dict) -> dict:
    retries = 0
    model_name = model["name"]

    try:
        mime_type, data_url = encode_image_as_data_url(image_path)
    except ValueError as e:
        return {
            "description": f"ERROR: {e}",
            "cost": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }

    while retries < MAX_RETRIES:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are an assistant that describes images."},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": data_url}}
                    ]}
                ],
                timeout=900.0
            )

            usage = response.usage
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens
            total_tokens = usage.total_tokens

            cost = (
                prompt_tokens / 1000 * model["price_input"] +
                completion_tokens / 1000 * model["price_output"]
            )

            return {
                "description": response.choices[0].message.content,
                "cost": cost,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            }

        except OpenAIError as e:
            retries += 1
            print(f"[ERROR] {model_name} on {image_path}: {e} (retry {retries})")
            time.sleep(RETRY_DELAY_BASE * (2 ** retries))

    return {
        "description": "ERROR: Max retries exceeded",
        "cost": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0
    }


def run_batch():
    results = []

    for filename in os.listdir(IMAGE_FOLDER):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".gif")):
            print(f"[SKIP] Unsupported extension: {filename}")
            continue

        image_path = os.path.join(IMAGE_FOLDER, filename)

        for model in MODELS:
            model_name = model["name"]
            print(f"[{model_name}] Processing {filename}...")

            result_data = describe_image(image_path, model)
            cost_tracker[model_name] += result_data["cost"]

            result_entry = {
                "image": filename,
                "model": model_name,
                "description": result_data["description"],
                "prompt_tokens": result_data["prompt_tokens"],
                "completion_tokens": result_data["completion_tokens"],
                "total_tokens": result_data["total_tokens"],
                "cost_usd": round(result_data["cost"], 6)
            }

            results.append(result_entry)

            with open(OUTPUT_LOG, "a") as f:
                f.write(json.dumps(result_entry) + "\n")

    print("\n--- Cost Summary ---")
    for model_name, total_cost in cost_tracker.items():
        print(f"{model_name}: ${total_cost:.4f}")


if __name__ == "__main__":
    run_batch()
