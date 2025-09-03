import logging
from typing import List, Optional

from scripts.chunking.models import Chunk  # Assuming Chunk model is here
from scripts.utils.logger import LoggerManager

logger = LoggerManager.get_logger("prompt")

DEFAULT_PROMPT_TEMPLATE = """
You are an expert assistant helping library systems librarians troubleshoot,
configure, and integrate tools like Alma, Primo, SAML authentication, APIs,
and other library technologies.

Your job is to answer practical questions based ONLY on the provided context.
If the context does not include the answer, clearly state that.

Use citations to support your answer, referring to sources using their IDs
(e.g., [doc_id_1], [doc_id_2]).

If the user's question is in Hebrew, please answer in Hebrew. Otherwise,
answer in the same language as the question.

---

Context:
{context_str}

---

User Question:
{query_str}

---

Answer:
"""


DEFAULT_PROMPT_TEMPLATE_V2 = """
You are an expert assistant helping library systems librarians troubleshoot, configure, and integrate tools like Alma, Primo, SAML authentication, APIs, and other library technologies.

Your job is to answer practical questions based ONLY on the provided context.
If the context does not contain the answer, clearly state that.

When forming your answer:

1. Start with “When or why to perform this task” — include prerequisites, decision criteria, or situations where this procedure is needed.
2. Provide a clear, step-by-step guide with numbered steps.
  2.1 For each major step, add a brief purpose or explanation (why the step matters).
3. Include at least one concrete example — *only if the context contains one* (e.g., specific collection names or scenarios). If no example exists in the context, omit this part entirely and do not invent one.
4. Add a “Tips & Pitfalls” section if the context contains warnings, best practices, or common mistakes to avoid.
5. If there are different options or methods in the context (e.g., different ways to add portfolios), briefly explain when to use each.
6. Maintain a training-friendly tone so new staff can follow the instructions without prior experience.
7. Use citations to support your answer, referring to sources using their IDs (e.g., [doc_id_1], [doc_id_2]).

If the context contains specific examples or case studies, include one.
If the context contains tips, pitfalls, or best practices, present them as a complete list. 
If multiple methods are listed for a step, explain when to use each method. 
Do not invent information — only use what is present in the context.

If the user's question is in Hebrew, answer in Hebrew. Otherwise, answer in the same language as the question.
---

Context:
{context_str}

---

User Question:
{query_str}

---

Answer:
"""

class PromptBuilder:
    """
    Builds prompts for the LMM by combining a user query with retrieved
    context chunks.
    """

    def __init__(self, template: str | None = None, run_id: Optional[str] = None):
        """
        Initializes the PromptBuilder.

        Args:
            template (str, optional): A custom prompt template.
                If None, DEFAULT_PROMPT_TEMPLATE is used.
                Must contain {context_str} and {query_str} placeholders.
        """
        self.run_id = run_id
        self.logger = LoggerManager.get_logger("prompt", run_id=run_id)
        self.template = template or DEFAULT_PROMPT_TEMPLATE_V2
        if ("{context_str}" not in self.template or
                "{query_str}" not in self.template):
            self.logger.error(
                "Prompt template must include {context_str} and "
                "{query_str} placeholders.",
                extra={"run_id": run_id} if run_id else {}
            )
            raise ValueError(
                "Prompt template must include {context_str} and "
                "{query_str} placeholders."
            )
        self.logger.debug("PromptBuilder initialized.", extra={"run_id": run_id} if run_id else {})

    def build_prompt(self, query: str, context_chunks: List[Chunk]) -> str:
        """
        Builds a complete prompt string.

        Args:
            query (str): The user's query.
            context_chunks (List[Chunk]): A list of context chunks retrieved
                from the RAG system.

        Returns:
            str: The fully formatted prompt string.
        """
        if not context_chunks:
            self.logger.warning("Building prompt with no context chunks.", extra={"run_id": self.run_id} if self.run_id else {})
            context_str = "No context provided."
        else:
            context_items = []
            for i, chunk in enumerate(context_chunks):
                # Handle doc_id safely
                source_id = (
                    chunk.meta.get("source_filepath") or
                    getattr(chunk, "doc_id", "unknown")
                )
                source_id_str = str(source_id).replace("\n", " ").strip()

                # Use text or description depending on chunk type
                text = (
                    getattr(chunk, "text", None) or
                    getattr(chunk, "description", None)
                )
                if not text:
                    self.logger.warning(
                        f"Skipping chunk with no usable content: "
                        f"{getattr(chunk, 'id', 'N/A')}",
                        extra={"run_id": self.run_id, "chunk_id": getattr(chunk, 'id', 'N/A')} if self.run_id else {"chunk_id": getattr(chunk, 'id', 'N/A')}
                    )
                    continue

                context_item = f"Source ID: [{source_id_str}]\nContent: {text}"

                # Add other relevant metadata if available, e.g., page number
                page_number = chunk.meta.get('page_number')
                if page_number:
                    context_item += f"\nPage: {page_number}"

                context_items.append(context_item)
            context_str = "\n\n---\n\n".join(context_items)

        self.logger.debug(
            f"Building prompt with {len(context_chunks)} context chunks",
            extra={"run_id": self.run_id, "query_length": len(query), "context_chunk_count": len(context_chunks)} if self.run_id else {"query_length": len(query), "context_chunk_count": len(context_chunks)}
        )

        # Replace placeholders in the template
        try:
            final_prompt = self.template.format(
                context_str=context_str, query_str=query
            )
        except KeyError as e:
            self.logger.error(
                f"Missing placeholder in prompt template: {e}. Ensure template "
                f"has {{context_str}} and {{query_str}}.",
                extra={"run_id": self.run_id, "missing_placeholder": str(e)} if self.run_id else {"missing_placeholder": str(e)},
                exc_info=True
            )
            # Fallback or re-raise, depending on desired robustness
            raise ValueError(
                f"Failed to format prompt template due to missing "
                f"placeholder: {e}"
            )

        self.logger.debug(f"Generated prompt length: {len(final_prompt)} characters", extra={"run_id": self.run_id, "prompt_length": len(final_prompt)} if self.run_id else {"prompt_length": len(final_prompt)})
        return final_prompt


if __name__ == '__main__':
    # Example Usage
    logging.basicConfig(level=logging.INFO)

    # Mock Chunks
    mock_chunks_data = [
        {
            "doc_id": "doc1.txt",
            "text": "The sky is blue.",
            "meta": {"source_filepath": "docs/doc1.txt", "page_number": 1},
            "token_count": 4,
        },
        {
            "doc_id": "doc2.pdf",
            "text": "An apple a day keeps the doctor away.",
            "meta": {"source_filepath": "pdfs/doc2.pdf"},
            "token_count": 8,
        },
        {
            "doc_id": "doc3.txt",
            "text": "Water is essential for life.",
            "meta": {"source_filepath": "notes/doc3.txt", "page_number": 5},
            "token_count": 6,
        },
    ]

    context_chunks = [Chunk(**data) for data in mock_chunks_data]

    user_query = "What color is the sky and why is water important?"

    logger.info("Starting PromptBuilder direct test...")
    try:
        # Test with default template
        builder_default = PromptBuilder()
        prompt_default = builder_default.build_prompt(user_query, context_chunks)
        logger.info(
            f"\n--- Generated Prompt (Default Template) ---\n"
            f"{prompt_default}\n----------------------------------------"
        )

        # Test with a custom template
        CUSTOM_TEMPLATE = """
        Contextual Information:
        ***
        {context_str}
        ***

        Based on the information above, please answer the question:
        {query_str}
        Remember to cite your sources as [Source ID string].
        """
        builder_custom = PromptBuilder(template=CUSTOM_TEMPLATE)
        prompt_custom = builder_custom.build_prompt(user_query, context_chunks)
        logger.info(
            f"\n--- Generated Prompt (Custom Template) ---\n"
            f"{prompt_custom}\n---------------------------------------"
        )

        # Test with no context
        prompt_no_context = builder_default.build_prompt("What is your name?", [])
        logger.info(
            f"\n--- Generated Prompt (No Context) ---\n"
            f"{prompt_no_context}\n------------------------------------"
        )

        # Test template validation (missing placeholder)
        try:
            # Missing {context_str}
            PromptBuilder(template="Query: {query_str}")
        except ValueError as e:
            logger.info(
                f"Successfully caught expected error for bad template: {e}"
            )

    except Exception as e:
        logger.error(
            f"An error occurred during PromptBuilder test: {e}",
            exc_info=True
        )

    logger.info("PromptBuilder direct test finished.")
