# Agents Module

This module contains AI agents that perform specialized tasks to enrich the data in the RAG pipeline. These agents are designed to be modular and can be selectively enabled or disabled through the project configuration.

## Core Components

- **`base.py`**: Defines the `AgentProtocol`, an abstract base class that provides a common interface for all agents. It mandates a `run` method that takes a `Chunk` and a `ProjectManager` as input and returns an enriched `Chunk` or a list of new chunks.

- **`image_insight_agent.py`**: Implements the `ImageInsightAgent`, a powerful agent that uses a multimodal LLM to generate descriptions for images found in the documents.

## ImageInsightAgent

The `ImageInsightAgent` is designed to enhance the retrieval process by providing textual descriptions for images. This allows the system to find relevant information even when it's contained within an image.

### Functionality

- **Image Analysis**: For each chunk that contains one or more images, the agent sends each image to a multimodal LLM (e.g., GPT-4o) along with a configurable prompt and the surrounding text from the chunk.
- **Description Generation**: The LLM analyzes the image in the context of the text and generates a detailed description of what the image shows.
- **Output Modes**: The agent can be configured with different output modes:
    - **`append_to_chunk` (default)**: The generated image descriptions are appended to the metadata of the original text chunk.
    - **`separate_chunk`**: The agent creates a new `ImageChunk` for each image, containing the generated description and metadata linking it back to the original text chunk.

### Configuration

The behavior of the `ImageInsightAgent` is controlled by the `agents` section in the `config.yml` file:

- **`enable_image_insight`**: A boolean to enable or disable the agent.
- **`image_agent_model`**: The name of the multimodal LLM to use (e.g., `gpt-4o`).
- **`image_prompt`**: The prompt template to use for generating the image descriptions. It can include a `{{ context }}` placeholder, which will be replaced with the text surrounding the image.
- **`output_mode`**: The output mode to use (`append_to_chunk` or `separate_chunk`).

## Usage

Agents are typically run as part of a larger pipeline, after the initial chunking stage. The following is a conceptual example of how the `ImageInsightAgent` might be used:

```python
from scripts.agents.image_insight_agent import ImageInsightAgent
from scripts.core.project_manager import ProjectManager

# Assuming 'project' is an initialized ProjectManager object
# and 'chunks' is a list of Chunk objects from the chunker.

if project.config.get("agents", {}).get("enable_image_insight", False):
    agent = ImageInsightAgent(project)
    enriched_chunks = []
    for chunk in chunks:
        # The agent returns a list, which may contain the original chunk
        # (potentially modified) and/or new chunks.
        result = agent.run(chunk, project)
        enriched_chunks.extend(result)

    # 'enriched_chunks' now contains the original chunks with added image
    # descriptions in their metadata, or additional ImageChunk objects.
```
