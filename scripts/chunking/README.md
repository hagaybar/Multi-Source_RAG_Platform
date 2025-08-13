# Chunking Module

This module is responsible for splitting raw documents into smaller, more manageable chunks. This is a critical step in the RAG pipeline, as it prepares the data for efficient embedding and retrieval. The chunking process is driven by a set of configurable rules that allow for different strategies to be applied to different document types.

## Core Components

- **`chunker_v3.py`**: This is the main chunking engine. It contains the `split` function, which takes the text and metadata of a raw document and returns a list of `Chunk` objects. It uses a variety of strategies to first split the document into smaller pieces and then merge them into chunks of an appropriate size.
- **`models.py`**: Defines the data models for the chunking process:
    - **`Chunk`**: Represents a single chunk of text. It includes the chunk's content, metadata, token count, and a unique ID.
    - **`Doc`**: Represents a full document as a collection of chunks.
    - **`ImageChunk`**: A specialized dataclass for representing image-based content, containing an ID, a description, and metadata.
- **`rules_v3.py`**: This module is responsible for loading, validating, and providing access to the chunking rules defined in the `configs/chunk_rules.yaml` file.

## Chunking Rules

The chunking process is governed by a set of rules defined in `configs/chunk_rules.yaml`. Each rule is associated with a specific `doc_type` and specifies the following:

- **`strategy`**: The method to use for initially splitting the document into smaller pieces.
- **`min_tokens`**: The minimum number of tokens a chunk should have.
- **`max_tokens`**: The maximum number of tokens a chunk can have.
- **`overlap`**: The number of tokens to overlap between consecutive chunks.

The `rules_v3.py` module loads these rules, validates them to ensure they contain all the required keys, and provides a `get_rule` function that retrieves the appropriate `ChunkRule` for a given `doc_type`.

## Chunking Strategies

The `chunker_v3.py` script implements several chunking strategies, which are selected based on the rule for the current `doc_type`:

- **`by_paragraph`**: Splits the text based on one or more blank lines.
- **`by_slide`**: Splits the text based on a `---` separator, which is used to delineate slides in a PowerPoint presentation.
- **`split_on_sheets`**: Treats the entire text of a spreadsheet as a single item.
- **`blank_line`**: Splits the text by double newlines.
- **`split_on_rows`**: Splits the text by newlines, treating each line as a row from a CSV file.
- **`by_email_block`**: Uses `spaCy` to split the text into sentences, which is suitable for email content.

## Workflow

1.  The `split` function in `chunker_v3.py` is called with the text and metadata of a raw document.
2.  It retrieves the appropriate `ChunkRule` for the document's `doc_type` using the `get_rule` function from `rules_v3.py`.
3.  For email documents, it first cleans the text by removing quoted replies and signatures using the `clean_email_text` utility.
4.  It then uses the specified `strategy` to split the text into a list of smaller strings (e.g., paragraphs, sentences).
5.  The `merge_chunks_with_overlap` function takes this list of strings and intelligently merges them into `Chunk` objects that respect the `min_tokens` and `max_tokens` constraints of the rule, while also adding the specified `overlap`.
6.  The function returns a list of `Chunk` objects, which are then ready to be embedded.
