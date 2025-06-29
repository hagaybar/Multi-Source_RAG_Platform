import logging
from typing import List
from scripts.chunking.models import Chunk # Assuming Chunk model is here

logger = logging.getLogger(__name__)

DEFAULT_PROMPT_TEMPLATE = """
You are a helpful AI assistant. Answer the user's query based ONLY on the provided context.
If the context does not contain the answer, state that the answer is not found in the provided context.
Cite the sources used to answer the query. Refer to sources using their IDs (e.g., [doc_id_1], [doc_id_2]).

Context:
{context_str}

Query: {query_str}

Answer:
"""

class PromptBuilder:
    """
    Builds prompts for the LMM by combining a user query with retrieved context chunks.
    """
    def __init__(self, template: str | None = None):
        """
        Initializes the PromptBuilder.

        Args:
            template (str, optional): A custom prompt template.
                                      If None, DEFAULT_PROMPT_TEMPLATE is used.
                                      Must contain {context_str} and {query_str} placeholders.
        """
        self.template = template or DEFAULT_PROMPT_TEMPLATE
        if "{context_str}" not in self.template or "{query_str}" not in self.template:
            logger.error("Prompt template must include {context_str} and {query_str} placeholders.")
            raise ValueError("Prompt template must include {context_str} and {query_str} placeholders.")
        logger.info("PromptBuilder initialized.")

    def build_prompt(self, query: str, context_chunks: List[Chunk]) -> str:
        """
        Builds a complete prompt string.

        Args:
            query (str): The user's query.
            context_chunks (List[Chunk]): A list of context chunks retrieved from the RAG system.

        Returns:
            str: The fully formatted prompt string.
        """
        if not context_chunks:
            logger.warning("Building prompt with no context chunks.")
            context_str = "No context provided."
        else:
            context_items = []
            for i, chunk in enumerate(context_chunks):
                # Prefer 'source_filepath' if available, else 'doc_id'
                source_id = chunk.meta.get('source_filepath', chunk.doc_id)
                # Ensure source_id is a string and usable in the prompt.
                # Replace problematic characters if necessary, or ensure they are clean upstream.
                source_id_str = str(source_id).replace("\n", " ").strip()

                context_item = f"Source ID: [{source_id_str}]\nContent: {chunk.text}"

                # Add other relevant metadata if available, e.g., page number
                page_number = chunk.meta.get('page_number')
                if page_number:
                    context_item += f"\nPage: {page_number}"

                context_items.append(context_item)
            context_str = "\n\n---\n\n".join(context_items)

        logger.info(f"Building prompt for query: '{query}' with {len(context_chunks)} context chunks.")

        # Replace placeholders in the template
        try:
            final_prompt = self.template.format(context_str=context_str, query_str=query)
        except KeyError as e:
            logger.error(f"Missing placeholder in prompt template: {e}. Ensure template has {{context_str}} and {{query_str}}.")
            # Fallback or re-raise, depending on desired robustness
            raise ValueError(f"Failed to format prompt template due to missing placeholder: {e}")

        logger.debug(f"Generated prompt: {final_prompt}")
        return final_prompt

if __name__ == '__main__':
    # Example Usage
    logging.basicConfig(level=logging.INFO)

    # Mock Chunks
    mock_chunks_data = [
        {"doc_id": "doc1.txt", "text": "The sky is blue.", "meta": {"source_filepath": "docs/doc1.txt", "page_number": 1}, "token_count": 4},
        {"doc_id": "doc2.pdf", "text": "An apple a day keeps the doctor away.", "meta": {"source_filepath": "pdfs/doc2.pdf"}, "token_count": 8},
        {"doc_id": "doc3.txt", "text": "Water is essential for life.", "meta": {"source_filepath": "notes/doc3.txt", "page_number": 5}, "token_count": 6},
    ]

    context_chunks = [Chunk(**data) for data in mock_chunks_data]

    user_query = "What color is the sky and why is water important?"

    logger.info("Starting PromptBuilder direct test...")
    try:
        # Test with default template
        builder_default = PromptBuilder()
        prompt_default = builder_default.build_prompt(user_query, context_chunks)
        logger.info(f"\n--- Generated Prompt (Default Template) ---\n{prompt_default}\n----------------------------------------")

        # Test with a custom template
        CUSTOM_TEMPLATE = """
        Contextual Information:
        ***
        {context_str}
        ***

        Based on the information above, please answer the question: {query_str}
        Remember to cite your sources as [Source ID string].
        """
        builder_custom = PromptBuilder(template=CUSTOM_TEMPLATE)
        prompt_custom = builder_custom.build_prompt(user_query, context_chunks)
        logger.info(f"\n--- Generated Prompt (Custom Template) ---\n{prompt_custom}\n---------------------------------------")

        # Test with no context
        prompt_no_context = builder_default.build_prompt("What is your name?", [])
        logger.info(f"\n--- Generated Prompt (No Context) ---\n{prompt_no_context}\n------------------------------------")

        # Test template validation (missing placeholder)
        try:
            PromptBuilder(template="Query: {query_str}") # Missing {context_str}
        except ValueError as e:
            logger.info(f"Successfully caught expected error for bad template: {e}")

    except Exception as e:
        logger.error(f"An error occurred during PromptBuilder test: {e}", exc_info=True)

    logger.info("PromptBuilder direct test finished.")
