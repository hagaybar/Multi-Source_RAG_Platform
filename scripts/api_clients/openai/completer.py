import os
import litellm
import logging

# Initialize logger for this module
logger = logging.getLogger(__name__)


class OpenAICompleter:
    """
    A client for getting completions from OpenAI models via LiteLLM.
    """

    def __init__(self, api_key: str | None = None, model_name: str = "gpt-3.5-turbo"):
        """
        Initializes the OpenAICompleter.
        LiteLLM will use the OPEN_AI environment variable by default if api_key is
        not provided.

        Args:
            api_key (str, optional): OpenAI API key. If provided, it will be used.
                                     Otherwise, LiteLLM will look for OPEN_AI env var.
            model_name (str, optional): Default OpenAI model to use for completions.
                                        Defaults to "gpt-3.5-turbo".
                                        Ensure this model name is prefixed with
                                        "openai/" 
                                        if required by your LiteLLM setup,
                                        though often not necessary for direct
                                        OpenAI calls.
                                        For direct OpenAI, "gpt-3.5-turbo" is fine.
        """
        if api_key:
            os.environ["OPEN_AI"] = api_key  # LiteLLM picks this up

        # Check if the API key is available for LiteLLM
        if not os.getenv("OPEN_AI"):
            logger.error(
                "OpenAI API key not found. Please set the OPEN_AI "
                "environment variable or pass api_key to constructor."
            )
            raise ValueError(
                "OpenAI API key not found. Please set the OPEN_AI "
                "environment variable or pass api_key to constructor."
            )

        # The model name for LiteLLM should be just the model ID, e.g.,
        # "gpt-3.5-turbo" for OpenAI
        self.model_name = model_name
        logger.info(
            f"OpenAICompleter initialized, will use LiteLLM for model: {self.model_name}"
        )

    def get_completion(
        self,
        prompt: str,
        model_name: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> str | None:
        """
        Gets a completion using LiteLLM.

        Args:
            prompt (str): The prompt to send to the model.
            model_name (str, optional): The model to use. If None, uses the 
                                        instance's default model.
                                        For OpenAI, this would be e.g. 
                                        "gpt-3.5-turbo", "gpt-4", etc.
            temperature (float, optional): Sampling temperature. Defaults to 0.7.
            max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 500.

        Returns:
            str | None: The content of the completion, or None if an error occurs.
        """
        current_model = model_name or self.model_name
        # For LiteLLM, ensure the model name is correctly formatted 
        # if it needs a prefix like "openai/"
        # However, for direct OpenAI calls through LiteLLM, "gpt-3.5-turbo" is typically sufficient.
        # If using a proxy or router, the model name might need to be "openai/gpt-3.5-turbo".
        # For simplicity, we assume direct usage here.

        logger.info(
            f"Requesting completion from LiteLLM for model: {current_model} "
            f"with prompt (first 100 chars): '{prompt[:100]}...'"
        )

        messages = [{"role": "user", "content": prompt}]

        try:
            response = litellm.completion(
                model=current_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=os.getenv("OPEN_AI"),
                # api_key is not explicitly passed here if OPEN_AI is set,
                # LiteLLM handles it. If self.api_key was set in __init__ from param,
                # it's already in os.environ.
            )

            if (
                response.choices
                and response.choices[0].message
                and response.choices[0].message.content
            ):
                content = response.choices[0].message.content
                logger.info(
                    f"Completion received successfully via LiteLLM. Length: {len(content)} chars."
                )
                return content
            else:
                logger.warning("No completion content received from LiteLLM or content is empty.")
                return None
        except litellm.exceptions.APIError as e:  # Catch LiteLLM specific API errors
            logger.error(f"LiteLLM API error: {e}")
            return None
        except Exception as e:  # Catch any other exception
            logger.error(f"An unexpected error occurred while getting completion via LiteLLM: {e}")
            return None

    def get_multimodal_completion(
        self,
        prompt: str,
        image_b64: str,
        model_name: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> str | None:
        """Sends a multimodal prompt (text + image) to the model via LiteLLM."""

        current_model = model_name or self.model_name
        logger.info(f"Requesting multimodal completion from LiteLLM for model: {current_model}")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                ],
            }
        ]

        try:
            response = litellm.completion(
                model=current_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=os.getenv("OPEN_AI"),
            )

            if (
                response.choices
                and response.choices[0].message
                and response.choices[0].message.content
            ):
                content = response.choices[0].message.content
                logger.info(
                    f"Multimodal completion received successfully via LiteLLM. "
                    f"Length: {len(content)} chars."
                )
                return content
            logger.warning("No completion content received from LiteLLM or content is empty.")
            return None
        except litellm.exceptions.APIError as e:
            logger.error(f"LiteLLM API error: {e}")
            return None
        except Exception as e:
            logger.error(
                f"An unexpected error occurred while getting multimodal completion via LiteLLM: {e}"
            )
            return None


if __name__ == '__main__':
    # Example Usage (requires OPEN_AI environment variable to be set)
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting OpenAICompleter (via LiteLLM) direct test...")

    # OPEN_AI should be set in the environment
    if not os.getenv("OPEN_AI"):
        logger.error("Cannot run test: OPEN_AI environment variable not set.")
    else:
        try:
            # No need to pass api_key if OPEN_AI is set
            completer = OpenAICompleter(model_name="gpt-3.5-turbo")
            test_prompt = "What is the capital of France? Respond in one sentence."
            logger.info(f"Sending test prompt: '{test_prompt}'")

            completion_content = completer.get_completion(
                prompt=test_prompt, temperature=0.5, max_tokens=50
            )

            if completion_content:
                logger.info(f"Test completion received: {completion_content}")
            else:
                logger.error("Test completion failed or returned empty content.")

        except ValueError as ve:
            logger.error(f"Initialization error during test: {ve}")
        except Exception as e:
            logger.error(f"An unexpected error occurred during the test: {e}")

    logger.info("OpenAICompleter (via LiteLLM) direct test finished.")
