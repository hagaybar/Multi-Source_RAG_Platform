# Utility Scripts

The `scripts/utils` directory provides helper modules used throughout the RAG platform. These small utilities handle common tasks such as configuration loading, logging, and file conversions.

- `__init__.py` – marks the folder as a Python package.
- `chunk_utils.py` – functions to deduplicate `Chunk` objects and load chunks from TSV files.
- `config_loader.py` – `ConfigLoader` for reading YAML configs with dot‑notation access.
- `create_demo_pptx.py` – generates a simple PowerPoint file used in tests.
- `email_utils.py` – `clean_email_text` to strip quoted replies, signatures and reply blocks.
- `image_utils.py` – utilities for saving images, caching them on disk and creating filenames.
- `logger.py` – `LoggerManager` and `JsonLogFormatter` providing configurable console/file logging.
- `msg2email.py` – converts Outlook `.msg` files into standard `.eml` format.

These helpers support higher‑level modules like the ingestion, embedding and retrieval components.
