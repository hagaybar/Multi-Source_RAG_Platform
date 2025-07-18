# Configs Folder

The `configs` folder is used to store configuration files for the project.

- `__init__.py`: This file is empty and is used to mark the `configs` folder as a Python package.
- `chunk_rules.yaml`: This YAML file defines the rules for chunking different types of documents.
    - For each document type (e.g., `email`, `docx`, `pdf`), it specifies:
        - `split_strategy`: The method to use for splitting the document (e.g., `split_on_blank_lines`, `split_on_headings`).
        - `min_chunk_size`: The minimum desired size for each chunk.
        - `notes`: Additional information or considerations for processing that document type.
    - For `pptx` (PowerPoint presentations), it also includes `token_bounds` and `overlap` parameters.
- `chunk_rules_old.yaml`: This YAML file contains older or deprecated chunking rules.

The `chunk_rules.yaml` file is crucial for the document processing pipeline, as it allows for customized chunking behavior based on file type, ensuring that the content is divided into meaningful segments for further processing (e.g., embedding and retrieval).
\n*Configuration examples are kept here for easy reference.*
