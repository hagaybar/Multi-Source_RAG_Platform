#!/usr/bin/env python3
"""
Pipeline Validator - Pre-flight checks for config changes

Prevents data corruption by validating changes before pipeline execution.
Shows user what will happen when config changes are detected.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class ValidationIssue:
    """Represents a validation issue."""
    level: str  # 'error', 'warning', 'info'
    category: str  # e.g., 'chunking', 'embedding', 'data'
    message: str
    old_value: Optional[str] = None
    new_value: Optional[str] = None
    recommendation: Optional[str] = None
    impact: Optional[str] = None


@dataclass
class ValidationReport:
    """Validation report containing all issues."""
    errors: List[ValidationIssue] = field(default_factory=list)
    warnings: List[ValidationIssue] = field(default_factory=list)
    info: List[ValidationIssue] = field(default_factory=list)

    def add_error(self, **kwargs):
        """Add an error issue."""
        self.errors.append(ValidationIssue(level='error', **kwargs))

    def add_warning(self, **kwargs):
        """Add a warning issue."""
        self.warnings.append(ValidationIssue(level='warning', **kwargs))

    def add_info(self, **kwargs):
        """Add an info issue."""
        self.info.append(ValidationIssue(level='info', **kwargs))

    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return len(self.errors) > 0

    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return len(self.warnings) > 0

    def is_empty(self) -> bool:
        """Check if report is empty."""
        return len(self.errors) == 0 and len(self.warnings) == 0 and len(self.info) == 0


class PipelineValidator:
    """Validates pipeline changes and shows impact."""

    # Known embedding model dimensions
    MODEL_DIMENSIONS = {
        'text-embedding-3-large': 3072,
        'text-embedding-3-small': 1536,
        'text-embedding-ada-002': 1536,
        'text-embedding-ada-001': 1024,
    }

    def __init__(self, project):
        """
        Initialize validator.

        Args:
            project: ProjectManager instance
        """
        self.project = project
        self.last_config_path = self.project.root_dir / "output" / ".last_config.yml"
        self.last_metadata_path = self.project.root_dir / "output" / ".last_metadata.json"

    def validate(self) -> ValidationReport:
        """
        Validate current config against last run.

        Returns:
            ValidationReport with errors, warnings, and info
        """
        report = ValidationReport()

        # Check if this is first run
        if not self.last_config_path.exists():
            report.add_info(
                category='first_run',
                message='First pipeline run for this project',
                impact='All data will be processed from scratch'
            )
            return report

        # Load last config
        with open(self.last_config_path, 'r') as f:
            last_config = yaml.safe_load(f)

        current_config = self.project.config

        # Load last metadata if exists
        last_metadata = {}
        if self.last_metadata_path.exists():
            with open(self.last_metadata_path, 'r') as f:
                last_metadata = json.load(f)

        # Validate each aspect
        self._validate_chunking(last_config, current_config, report)
        self._validate_embedding(last_config, current_config, report)
        self._validate_data_additions(last_metadata, report)
        self._validate_llm_config(last_config, current_config, report)

        return report

    def _validate_chunking(self, last_config: Dict, current_config: Dict,
                          report: ValidationReport):
        """Validate chunking configuration changes."""
        # Get chunking configs (from chunk_rules.yaml referenced in config)
        # For now, check if chunk_rules changed via file modification time
        chunk_rules_path = Path("configs/chunk_rules.yaml")

        if not chunk_rules_path.exists():
            return

        last_chunk_mtime = last_config.get('_metadata', {}).get('chunk_rules_mtime')
        current_chunk_mtime = chunk_rules_path.stat().st_mtime

        if last_chunk_mtime and current_chunk_mtime > last_chunk_mtime:
            # Chunk rules changed
            existing_chunks = list(self.project.get_input_dir().glob("chunks_*.tsv"))

            if existing_chunks:
                report.add_warning(
                    category='chunking',
                    message='Chunking rules have changed since last run',
                    old_value=f'Last modified: {last_chunk_mtime}',
                    new_value=f'Current modified: {current_chunk_mtime}',
                    impact='Will create chunks with different sizes, mixed with old chunks',
                    recommendation=(
                        'Delete existing chunks to avoid mixed sizes:\n'
                        f'  rm {self.project.get_input_dir()}/chunks_*.tsv'
                    )
                )

    def _validate_embedding(self, last_config: Dict, current_config: Dict,
                           report: ValidationReport):
        """Validate embedding configuration changes."""
        last_emb = last_config.get('embedding', {})
        current_emb = current_config.get('embedding', {})

        # Check model change
        last_model = last_emb.get('model')
        current_model = current_emb.get('model')

        if last_model and current_model and last_model != current_model:
            # Model changed - check dimensions
            last_dims = self.MODEL_DIMENSIONS.get(last_model, 'unknown')
            current_dims = self.MODEL_DIMENSIONS.get(current_model, 'unknown')

            if last_dims != current_dims:
                # Dimension mismatch - critical error
                existing_indices = list((self.project.root_dir / "output" / "faiss").glob("*.faiss"))

                if existing_indices:
                    report.add_error(
                        category='embedding',
                        message='Embedding model dimension mismatch',
                        old_value=f'{last_model} ({last_dims}d)',
                        new_value=f'{current_model} ({current_dims}d)',
                        impact='FAISS indices are incompatible with new model',
                        recommendation=(
                            'Delete existing indices and metadata:\n'
                            f'  rm -r {self.project.root_dir / "output" / "faiss"}/\n'
                            f'  rm -r {self.project.root_dir / "output" / "metadata"}/'
                        )
                    )
            else:
                # Same dimensions, just model change
                report.add_warning(
                    category='embedding',
                    message='Embedding model changed',
                    old_value=last_model,
                    new_value=current_model,
                    impact='Existing embeddings will be kept, new data will use new model',
                    recommendation='Consider re-embedding all data for consistency'
                )

        # Check provider change
        last_provider = last_emb.get('provider')
        current_provider = current_emb.get('provider')

        if last_provider and current_provider and last_provider != current_provider:
            report.add_warning(
                category='embedding',
                message='Embedding provider changed',
                old_value=last_provider,
                new_value=current_provider,
                impact='May affect embedding quality/compatibility'
            )

    def _validate_data_additions(self, last_metadata: Dict, report: ValidationReport):
        """Validate data additions."""
        # Count current raw files
        raw_dir = self.project.raw_docs_dir()
        current_files = []

        if raw_dir.exists():
            current_files = [f for f in raw_dir.rglob("*.*")
                           if f.is_file() and not f.name.startswith('.')]

        current_count = len(current_files)
        last_count = last_metadata.get('raw_file_count', 0)

        if current_count > last_count:
            added = current_count - last_count
            report.add_info(
                category='data',
                message=f'Added {added} new raw file(s)',
                old_value=f'{last_count} files',
                new_value=f'{current_count} files',
                impact='New files will be ingested, chunked, and embedded',
                recommendation=(
                    'Deduplication will skip unchanged content (via content_hash). '
                    'Only truly new chunks will be embedded.'
                )
            )
        elif current_count < last_count:
            removed = last_count - current_count
            report.add_warning(
                category='data',
                message=f'Removed {removed} raw file(s)',
                old_value=f'{last_count} files',
                new_value=f'{current_count} files',
                impact='Old chunks/embeddings from removed files will remain',
                recommendation=(
                    'Consider deleting old data if files were intentionally removed:\n'
                    f'  rm {self.project.get_input_dir()}/chunks_*.tsv\n'
                    f'  rm -r {self.project.root_dir / "output" / "faiss"}/\n'
                    f'  rm -r {self.project.root_dir / "output" / "metadata"}/'
                )
            )

    def _validate_llm_config(self, last_config: Dict, current_config: Dict,
                            report: ValidationReport):
        """Validate LLM configuration changes."""
        last_llm = last_config.get('llm', {})
        current_llm = current_config.get('llm', {})

        # Check model change
        last_model = last_llm.get('model')
        current_model = current_llm.get('model')

        if last_model and current_model and last_model != current_model:
            report.add_info(
                category='llm',
                message='LLM model changed',
                old_value=last_model,
                new_value=current_model,
                impact='Future queries will use new model'
            )

        # Check prompt strategy change
        last_strategy = last_llm.get('prompt_strategy')
        current_strategy = current_llm.get('prompt_strategy')

        if last_strategy and current_strategy and last_strategy != current_strategy:
            report.add_info(
                category='llm',
                message='Prompt strategy changed',
                old_value=last_strategy,
                new_value=current_strategy,
                impact='Query responses may differ'
            )

    def print_report(self, report: ValidationReport) -> bool:
        """
        Print validation report to console.

        Args:
            report: ValidationReport to print

        Returns:
            bool: True if safe to proceed (no errors), False otherwise
        """
        if report.is_empty():
            print("✅ No configuration changes detected - safe to proceed")
            return True

        print("=" * 80)
        print("PIPELINE VALIDATION REPORT")
        print("=" * 80)

        # Print errors (blocking)
        if report.has_errors():
            print("\n❌ CRITICAL ERRORS - Cannot proceed:\n")
            for err in report.errors:
                print(f"  [{err.category.upper()}] {err.message}")
                if err.old_value:
                    print(f"    Old: {err.old_value}")
                if err.new_value:
                    print(f"    New: {err.new_value}")
                if err.impact:
                    print(f"    Impact: {err.impact}")
                if err.recommendation:
                    print(f"    Fix: {err.recommendation}")
                print()

            print("=" * 80)
            print("❌ Pipeline execution BLOCKED - fix errors above first")
            print("=" * 80)
            return False

        # Print warnings (proceed with caution)
        if report.has_warnings():
            print("\n⚠️  WARNINGS - Proceed with caution:\n")
            for warn in report.warnings:
                print(f"  [{warn.category.upper()}] {warn.message}")
                if warn.old_value:
                    print(f"    Old: {warn.old_value}")
                if warn.new_value:
                    print(f"    New: {warn.new_value}")
                if warn.impact:
                    print(f"    Impact: {warn.impact}")
                if warn.recommendation:
                    print(f"    Recommendation: {warn.recommendation}")
                print()

        # Print info (FYI)
        if report.info:
            print("\n✓ INFORMATION:\n")
            for info in report.info:
                print(f"  [{info.category.upper()}] {info.message}")
                if info.old_value:
                    print(f"    Previous: {info.old_value}")
                if info.new_value:
                    print(f"    Current: {info.new_value}")
                if info.impact:
                    print(f"    Impact: {info.impact}")
                if info.recommendation:
                    print(f"    Note: {info.recommendation}")
                print()

        print("=" * 80)

        if report.has_warnings():
            print("⚠️  Warnings detected - review carefully before proceeding")
        else:
            print("✅ Safe to proceed (info messages only)")

        print("=" * 80)

        return True

    def save_current_state(self):
        """Save current config and metadata for next validation."""
        # Ensure output directory exists
        output_dir = self.project.root_dir / "output"
        output_dir.mkdir(exist_ok=True)

        # Save config snapshot
        config_to_save = dict(self.project.config)

        # Add metadata
        config_to_save['_metadata'] = {
            'saved_at': str(Path(__file__).stat().st_mtime),
            'chunk_rules_mtime': Path("configs/chunk_rules.yaml").stat().st_mtime if Path("configs/chunk_rules.yaml").exists() else None
        }

        with open(self.last_config_path, 'w') as f:
            yaml.dump(config_to_save, f, default_flow_style=False)

        # Save metadata
        raw_dir = self.project.raw_docs_dir()
        raw_files = []
        if raw_dir.exists():
            raw_files = [f for f in raw_dir.rglob("*.*")
                        if f.is_file() and not f.name.startswith('.')]

        metadata = {
            'raw_file_count': len(raw_files),
            'saved_at': str(Path(__file__).stat().st_mtime)
        }

        with open(self.last_metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)


def validate_pipeline(project) -> Tuple[bool, ValidationReport]:
    """
    Convenience function to validate pipeline.

    Args:
        project: ProjectManager instance

    Returns:
        Tuple of (can_proceed: bool, report: ValidationReport)
    """
    validator = PipelineValidator(project)
    report = validator.validate()
    can_proceed = validator.print_report(report)
    return can_proceed, report


if __name__ == "__main__":
    # Test the validator
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from scripts.core.project_manager import ProjectManager

    # Test on Primo_List_2
    project = ProjectManager(Path("data/projects/Primo_List_2"))

    print("Testing PipelineValidator...\n")
    can_proceed, report = validate_pipeline(project)

    if can_proceed:
        print("\nValidation passed - pipeline can proceed")
    else:
        print("\nValidation failed - pipeline blocked")
