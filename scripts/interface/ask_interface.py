from typing import List, Tuple
from scripts.core.project_manager import ProjectManager
from scripts.retrieval.retrieval_manager import RetrievalManager
from scripts.prompting.prompt_builder import PromptBuilder
from scripts.api_clients.openai.completer import OpenAICompleter


def run_ask(
    project_path: str,
    query: str,
    top_k: int = 5,
    model_name: str = "gpt-3.5-turbo",
    temperature: float = 0.7,
    max_tokens: int = 500,
) -> Tuple[str, List[str]]:
    """
    Executes the RAG ask pipeline given a project path and a query.

    Returns:
        answer (str): The generated answer.
        sources (List[str]): List of cited source identifiers.
    """
    project = ProjectManager(project_path)
    retriever = RetrievalManager(project)
    chunks = retriever.retrieve(query=query, top_k=top_k)

    prompt_builder = PromptBuilder()
    prompt = prompt_builder.build_prompt(query=query, context_chunks=chunks)

    completer = OpenAICompleter(model_name=model_name)
    answer = completer.get_completion(
        prompt=prompt, temperature=temperature, max_tokens=max_tokens
    )

    # Build clean source list
    sources = []
    for chunk in chunks:
        source = chunk.meta.get("source_filepath", chunk.doc_id)
        page = chunk.meta.get("page_number")
        if page:
            source += f", page {page}"
        sources.append(source)

    return answer, sources
