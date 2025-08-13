# Utility Scripts

The `scripts/utils` directory contains a collection of helper modules that provide essential services across the RAG platform. These utilities handle common tasks like configuration management, logging, file conversions, and UI rendering.

- `__init__.py`: Marks the directory as a Python package, allowing its modules to be imported elsewhere.
- `chunk_utils.py`: Provides functions for deduplicating `Chunk` objects and loading text chunks from TSV files.
- `config_loader.py`: Implements a `ConfigLoader` class for reading YAML configuration files with easy, dot-notation access.
- `create_demo_pptx.py`: A simple script that generates a basic PowerPoint presentation file, primarily used for testing purposes.
- `email_utils.py`: Includes the `clean_email_text` function, which strips quoted replies, signatures, and other noisy elements from email bodies.
- `image_utils.py`: Offers utilities for handling images, including saving, caching, and generating unique filenames.
- `logger.py`: Contains a `LoggerManager` and `JsonLogFormatter` for setting up configurable logging to both the console and files.
- `msg2email.py`: A utility to convert Microsoft Outlook `.msg` files into the standard `.eml` format.
- `run_logger.py`: Implements the `RunLogger` class, which logs the inputs and outputs of a pipeline run (e.g., retrieved chunks, prompts, and LLM responses) into a dedicated run-specific directory.
- `translation_utils.py`: Provides a `translate_to_english` function that uses the OpenAI API to translate text.
- `ui_utils.py`: Contains helper functions for rendering text in a Streamlit UI, with special handling for right-to-left (RTL) languages like Hebrew.

These utilities are fundamental to the operation of higher-level components, including the ingestion, embedding, and retrieval pipelines.
