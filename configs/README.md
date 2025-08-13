# Configurations

This directory stores configuration files that control the behavior of various components of the RAG platform.

## `chunk_rules.yaml`

This is the central configuration file for the chunking process. It defines a set of rules that determine how different types of documents are split into smaller chunks. The `scripts/chunking/rules_v3.py` module is responsible for loading and parsing this file.

### Structure

The file is a YAML dictionary where each key is a `doc_type` (e.g., `docx`, `pdf`, `eml`) and the value is a rule object with the following keys:

- **`strategy`**: The name of the strategy to use for splitting the document. This corresponds to one of the strategies implemented in `scripts/chunking/chunker_v3.py` (e.g., `by_paragraph`, `by_slide`, `split_on_rows`).
- **`min_tokens`**: The minimum number of tokens that a chunk should have.
- **`max_tokens`**: The maximum number of tokens that a chunk can have.
- **`overlap`**: The number of tokens to overlap between consecutive chunks.

A `default` rule is also defined, which is used as a fallback for any `doc_type` that does not have a specific rule.

### Purpose

By externalizing the chunking rules into a configuration file, we can easily modify the chunking behavior for different document types without having to change the code. This makes the platform more flexible and easier to maintain. For example, we can define a `by_paragraph` strategy for text-heavy documents like DOCX and PDF, and a `by_slide` strategy for presentations.
