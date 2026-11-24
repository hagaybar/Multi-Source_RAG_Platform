"""
Domain Knowledge Injector - Injects domain-specific context into prompts.

Part of Phase 6: Query Intelligence
Enriches prompts with relevant domain knowledge about Primo/library systems
to improve LLM understanding and answer quality.
"""

import yaml
from pathlib import Path
from typing import List, Dict, Optional, Set
from scripts.utils.logger import LoggerManager


class DomainKnowledgeInjector:
    """
    Injects domain knowledge into prompts based on detected entities.

    Workflow:
    1. Load domain knowledge from YAML file
    2. Detect entities in query (from QueryAnalysis)
    3. Look up relevant context for each entity
    4. Inject context into prompt

    Example:
        injector = DomainKnowledgeInjector()

        entities = ["Primo VE", "facets"]
        context = injector.get_context(entities)
        # Returns formatted context about Primo VE and facets

        prompt = injector.enrich_prompt(original_prompt, entities)
        # Returns prompt with domain context injected
    """

    def __init__(
        self,
        knowledge_path: Optional[Path] = None,
        run_id: Optional[str] = None
    ):
        """
        Initialize Domain Knowledge Injector.

        Args:
            knowledge_path: Path to domain knowledge YAML file
                           (default: data/knowledge/primo_domain_knowledge.yaml)
            run_id: Optional run ID for logging
        """
        if knowledge_path is None:
            # Default path
            project_root = Path(__file__).parent.parent.parent
            knowledge_path = project_root / "data" / "knowledge" / "primo_domain_knowledge.yaml"

        self.knowledge_path = Path(knowledge_path)
        self.logger = LoggerManager.get_logger("domain_knowledge_injector", run_id=run_id)
        self.knowledge = {}

        # Load knowledge base
        self._load_knowledge()

        self.logger.info(
            f"DomainKnowledgeInjector initialized with {len(self.knowledge.get('entities', {}))} entities"
        )

    def _load_knowledge(self):
        """Load domain knowledge from YAML file."""
        try:
            if not self.knowledge_path.exists():
                self.logger.warning(
                    f"Knowledge file not found: {self.knowledge_path}. "
                    f"Using empty knowledge base."
                )
                self.knowledge = {"entities": {}, "workflows": {}, "common_issues": {}}
                return

            with open(self.knowledge_path, 'r', encoding='utf-8') as f:
                self.knowledge = yaml.safe_load(f) or {}

            self.logger.info(
                f"Loaded knowledge base from {self.knowledge_path}: "
                f"{len(self.knowledge.get('entities', {}))} entities, "
                f"{len(self.knowledge.get('workflows', {}))} workflows, "
                f"{len(self.knowledge.get('common_issues', {}))} common issues"
            )

        except Exception as e:
            self.logger.error(f"Failed to load knowledge base: {e}", exc_info=True)
            self.knowledge = {"entities": {}, "workflows": {}, "common_issues": {}}

    def get_context(
        self,
        entities: List[str],
        include_workflows: bool = True,
        include_issues: bool = True
    ) -> str:
        """
        Get formatted context for a list of entities.

        Args:
            entities: List of entity names (e.g., ["Primo VE", "facets"])
            include_workflows: Whether to include relevant workflows
            include_issues: Whether to include common issues

        Returns:
            Formatted context string to inject into prompt
        """
        if not entities:
            return ""

        # Resolve aliases and expand entities
        resolved_entities = self._resolve_entities(entities)

        if not resolved_entities:
            return ""

        # Build context
        context_parts = []

        # Add entity context
        context_parts.append("**Domain Knowledge:**\n")
        for entity_name in sorted(resolved_entities):
            entity_info = self.knowledge.get("entities", {}).get(entity_name)
            if entity_info:
                entity_context = self._format_entity(entity_name, entity_info)
                context_parts.append(entity_context)

        # Add workflows if relevant
        if include_workflows:
            workflows = self._get_relevant_workflows(resolved_entities)
            if workflows:
                context_parts.append("\n**Relevant Workflows:**\n")
                for workflow_name, workflow_info in workflows.items():
                    workflow_context = self._format_workflow(workflow_name, workflow_info)
                    context_parts.append(workflow_context)

        # Add common issues if relevant
        if include_issues:
            issues = self._get_relevant_issues(resolved_entities)
            if issues:
                context_parts.append("\n**Common Issues:**\n")
                for issue_name, issue_info in issues.items():
                    issue_context = self._format_issue(issue_name, issue_info)
                    context_parts.append(issue_context)

        return "\n".join(context_parts)

    def enrich_prompt(
        self,
        original_prompt: str,
        entities: List[str],
        include_workflows: bool = True,
        include_issues: bool = True
    ) -> str:
        """
        Enrich a prompt with domain context.

        Args:
            original_prompt: The original prompt to enrich
            entities: List of detected entities
            include_workflows: Whether to include workflows
            include_issues: Whether to include common issues

        Returns:
            Enriched prompt with domain context injected
        """
        context = self.get_context(entities, include_workflows, include_issues)

        if not context:
            return original_prompt

        # Inject context before the main prompt
        enriched = f"{context}\n\n---\n\n{original_prompt}"

        self.logger.info(
            f"Enriched prompt with context for {len(entities)} entities "
            f"(+{len(enriched) - len(original_prompt)} chars)"
        )

        return enriched

    def _resolve_entities(self, entities: List[str]) -> Set[str]:
        """
        Resolve entity names from input (handle aliases and normalization).

        Args:
            entities: List of entity names or aliases

        Returns:
            Set of canonical entity names
        """
        resolved = set()
        entity_db = self.knowledge.get("entities", {})

        for entity in entities:
            entity_lower = entity.lower()

            # Direct match (case-insensitive)
            for canonical_name, info in entity_db.items():
                if canonical_name.lower() == entity_lower:
                    resolved.add(canonical_name)
                    # Add related entities
                    related = info.get("related_to", [])
                    resolved.update([r for r in related if r in entity_db])
                    break

                # Alias match
                aliases = info.get("aliases", [])
                if any(alias.lower() == entity_lower for alias in aliases):
                    resolved.add(canonical_name)
                    # Add related entities
                    related = info.get("related_to", [])
                    resolved.update([r for r in related if r in entity_db])
                    break

        return resolved

    def _format_entity(self, name: str, info: Dict) -> str:
        """Format entity information for prompt."""
        parts = [f"- **{name}**: {info.get('description', 'No description')}"]

        if info.get("context"):
            # Trim context to first 2-3 sentences
            context = info["context"].strip()
            sentences = context.split(". ")
            short_context = ". ".join(sentences[:2]) + ("." if len(sentences) > 2 else "")
            parts.append(f"  {short_context}")

        return "\n".join(parts)

    def _format_workflow(self, name: str, info: Dict) -> str:
        """Format workflow information for prompt."""
        parts = [f"- **{name}**:"]
        steps = info.get("steps", [])
        for i, step in enumerate(steps[:3], 1):  # Show first 3 steps
            parts.append(f"  {i}. {step}")
        if len(steps) > 3:
            parts.append(f"  ... ({len(steps) - 3} more steps)")
        return "\n".join(parts)

    def _format_issue(self, name: str, info: Dict) -> str:
        """Format common issue information for prompt."""
        parts = [f"- **{name}**:"]
        causes = info.get("causes", [])
        solutions = info.get("solutions", [])

        if causes:
            parts.append(f"  Causes: {', '.join(causes[:2])}")
        if solutions:
            parts.append(f"  Solutions: {', '.join(solutions[:2])}")

        return "\n".join(parts)

    def _get_relevant_workflows(self, entities: Set[str]) -> Dict:
        """Get workflows relevant to entities."""
        relevant = {}
        workflows = self.knowledge.get("workflows", {})

        for workflow_name, workflow_info in workflows.items():
            related = workflow_info.get("related_entities", [])
            if any(entity in entities for entity in related):
                relevant[workflow_name] = workflow_info

        return relevant

    def _get_relevant_issues(self, entities: Set[str]) -> Dict:
        """Get common issues relevant to entities."""
        relevant = {}
        issues = self.knowledge.get("common_issues", {})

        for issue_name, issue_info in issues.items():
            related = issue_info.get("entities", [])
            if any(entity in entities for entity in related):
                relevant[issue_name] = issue_info

        return relevant
