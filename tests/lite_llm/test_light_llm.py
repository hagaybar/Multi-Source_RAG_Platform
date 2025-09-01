# test_litellm_gpt5.py
import os
import litellm

# Optional: turn on full request/response debug
litellm._turn_on_debug()

# Make sure your API key is set
# Either export it in the shell: export OPEN_AI="sk-..."
# Or set it here:
# os.environ["OPEN_AI"] = "sk-..."

try:
    response = litellm.completion(
        model="gpt-5",  # Change to the exact model name in your config
        messages=[{"role": "user", "content": "Name the capital of France."}],
        # Keep it simple — avoid temperature if unsure it's supported
        # temperature=1,  # Uncomment only if model supports it
        max_completion_tokens=200  # Use max_completion_tokens for newer models
    )

    print("\n=== Raw LiteLLM Response ===")
    print(response)

    if hasattr(response, "choices") and response.choices:
        msg = getattr(response.choices[0].message, "content", None)
        print("\n=== Extracted Content ===")
        print(msg if msg else "[No content in message.content]")
    else:
        print("\n[No choices returned]")

except Exception as e:
    print(f"[ERROR] LiteLLM request failed: {e}")
