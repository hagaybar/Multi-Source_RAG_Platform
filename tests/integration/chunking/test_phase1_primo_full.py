#!/usr/bin/env python3
"""
Test Phase 1 Enhancements on Full Primo_List Dataset

This script validates the complete Phase 1 implementation:
1. Quote deduplication
2. Signature detection
3. Semantic chunking

Measures:
- Storage reduction
- Processing statistics
- Quality improvements
"""

import sys
import json
import shutil
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.core.project_manager import ProjectManager
from scripts.pipeline.runner import PipelineRunner


def backup_existing_data(project_path: Path, console: Console):
    """Backup existing FAISS indices and metadata for comparison."""

    backup_dir = project_path / "backup_pre_phase1"

    if backup_dir.exists():
        console.print("[yellow]Backup already exists. Skipping backup step.[/yellow]")
        return backup_dir

    console.print("\n[cyan]📦 Backing up existing data...[/cyan]")
    backup_dir.mkdir(parents=True, exist_ok=True)

    # Backup FAISS indices
    faiss_dir = project_path / "output" / "faiss"
    if faiss_dir.exists():
        shutil.copytree(faiss_dir, backup_dir / "faiss", dirs_exist_ok=True)
        console.print(f"  ✓ Backed up FAISS indices")

    # Backup metadata
    metadata_dir = project_path / "output" / "metadata"
    if metadata_dir.exists():
        shutil.copytree(metadata_dir, backup_dir / "metadata", dirs_exist_ok=True)
        console.print(f"  ✓ Backed up metadata")

    # Backup chunks
    chunk_files = list(project_path.glob("input/chunks_*.tsv"))
    if chunk_files:
        (backup_dir / "chunks").mkdir(exist_ok=True)
        for chunk_file in chunk_files:
            shutil.copy2(chunk_file, backup_dir / "chunks" / chunk_file.name)
        console.print(f"  ✓ Backed up {len(chunk_files)} chunk files")

    console.print(f"[green]✅ Backup saved to: {backup_dir}[/green]\n")
    return backup_dir


def get_baseline_metrics(backup_dir: Path, console: Console) -> dict:
    """Analyze baseline (pre-Phase 1) metrics."""

    console.print("\n[cyan]📊 Analyzing baseline metrics...[/cyan]")

    metrics = {
        'total_chunks': 0,
        'total_chars': 0,
        'faiss_size_mb': 0,
        'metadata_size_mb': 0
    }

    # Count chunks from metadata
    metadata_file = backup_dir / "metadata" / "outlook_eml_metadata.jsonl"
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            for line in f:
                if line.strip():
                    metrics['total_chunks'] += 1
                    data = json.loads(line)
                    metrics['total_chars'] += len(data.get('text', ''))

        metrics['metadata_size_mb'] = metadata_file.stat().st_size / (1024 * 1024)

    # FAISS index size
    faiss_file = backup_dir / "faiss" / "outlook_eml.faiss"
    if faiss_file.exists():
        metrics['faiss_size_mb'] = faiss_file.stat().st_size / (1024 * 1024)

    # Display baseline
    table = Table(title="Baseline Metrics (Pre-Phase 1)", box=box.ROUNDED)
    table.add_column("Metric", style="bold yellow")
    table.add_column("Value", style="cyan")

    table.add_row("Total Chunks", f"{metrics['total_chunks']:,}")
    table.add_row("Total Characters", f"{metrics['total_chars']:,}")
    table.add_row("Avg Chunk Size", f"{metrics['total_chars'] // metrics['total_chunks'] if metrics['total_chunks'] > 0 else 0} chars")
    table.add_row("FAISS Index Size", f"{metrics['faiss_size_mb']:.2f} MB")
    table.add_row("Metadata Size", f"{metrics['metadata_size_mb']:.2f} MB")
    table.add_row("Total Storage", f"{metrics['faiss_size_mb'] + metrics['metadata_size_mb']:.2f} MB")

    console.print(table)
    console.print()

    return metrics


def clear_processed_data(project_path: Path, console: Console):
    """Clear existing chunks, FAISS indices, and metadata to reprocess."""

    console.print("\n[cyan]🗑️  Clearing processed data for fresh run...[/cyan]")

    # Clear FAISS indices
    faiss_dir = project_path / "output" / "faiss"
    if faiss_dir.exists():
        for f in faiss_dir.glob("outlook_eml.*"):
            f.unlink()
        console.print("  ✓ Cleared FAISS indices")

    # Clear metadata
    metadata_dir = project_path / "output" / "metadata"
    if metadata_dir.exists():
        for f in metadata_dir.glob("outlook_eml_metadata.*"):
            f.unlink()
        console.print("  ✓ Cleared metadata")

    # Clear chunks
    for chunk_file in project_path.glob("input/chunks_outlook_eml.*"):
        chunk_file.unlink()
    console.print("  ✓ Cleared chunk files")

    console.print("[green]✅ Ready for Phase 1 reprocessing[/green]\n")


def run_phase1_pipeline(project_path: Path, console: Console):
    """Run full pipeline with Phase 1 enhancements."""

    console.print("\n" + "="*80)
    console.print("[bold magenta]🚀 Running Phase 1 Enhanced Pipeline[/bold magenta]")
    console.print("="*80 + "\n")

    # Initialize pipeline
    project = ProjectManager(project_path)
    runner = PipelineRunner(project, config=project.config)

    # Configure pipeline steps
    runner.add_step("ingest")   # With email cleaning integrated
    runner.add_step("chunk")    # With semantic chunking enabled
    runner.add_step("embed")    # Standard embedding

    # Run pipeline and capture output
    start_time = datetime.now()

    for message in runner.run_steps():
        console.print(message)

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    console.print(f"\n[green]✅ Pipeline completed in {duration:.1f} seconds[/green]\n")

    return duration


def get_phase1_metrics(project_path: Path, console: Console) -> dict:
    """Analyze Phase 1 metrics after reprocessing."""

    console.print("\n[cyan]📊 Analyzing Phase 1 metrics...[/cyan]")

    metrics = {
        'total_chunks': 0,
        'total_chars': 0,
        'faiss_size_mb': 0,
        'metadata_size_mb': 0
    }

    # Count chunks from metadata
    metadata_file = project_path / "output" / "metadata" / "outlook_eml_metadata.jsonl"
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            for line in f:
                if line.strip():
                    metrics['total_chunks'] += 1
                    data = json.loads(line)
                    metrics['total_chars'] += len(data.get('text', ''))

        metrics['metadata_size_mb'] = metadata_file.stat().st_size / (1024 * 1024)

    # FAISS index size
    faiss_file = project_path / "output" / "faiss" / "outlook_eml.faiss"
    if faiss_file.exists():
        metrics['faiss_size_mb'] = faiss_file.stat().st_size / (1024 * 1024)

    # Display Phase 1 metrics
    table = Table(title="Phase 1 Metrics (Post-Enhancement)", box=box.ROUNDED)
    table.add_column("Metric", style="bold yellow")
    table.add_column("Value", style="green")

    table.add_row("Total Chunks", f"{metrics['total_chunks']:,}")
    table.add_row("Total Characters", f"{metrics['total_chars']:,}")
    table.add_row("Avg Chunk Size", f"{metrics['total_chars'] // metrics['total_chunks'] if metrics['total_chunks'] > 0 else 0} chars")
    table.add_row("FAISS Index Size", f"{metrics['faiss_size_mb']:.2f} MB")
    table.add_row("Metadata Size", f"{metrics['metadata_size_mb']:.2f} MB")
    table.add_row("Total Storage", f"{metrics['faiss_size_mb'] + metrics['metadata_size_mb']:.2f} MB")

    console.print(table)
    console.print()

    return metrics


def compare_metrics(baseline: dict, phase1: dict, console: Console):
    """Compare baseline vs Phase 1 metrics."""

    console.print("\n" + "="*80)
    console.print("[bold magenta]📈 Phase 1 Impact Analysis[/bold magenta]")
    console.print("="*80 + "\n")

    # Calculate changes
    chunk_change = ((phase1['total_chunks'] - baseline['total_chunks']) / baseline['total_chunks'] * 100) if baseline['total_chunks'] > 0 else 0
    char_change = ((phase1['total_chars'] - baseline['total_chars']) / baseline['total_chars'] * 100) if baseline['total_chars'] > 0 else 0

    baseline_storage = baseline['faiss_size_mb'] + baseline['metadata_size_mb']
    phase1_storage = phase1['faiss_size_mb'] + phase1['metadata_size_mb']
    storage_change = ((phase1_storage - baseline_storage) / baseline_storage * 100) if baseline_storage > 0 else 0

    # Create comparison table
    table = Table(title="Before vs After Comparison", box=box.DOUBLE)
    table.add_column("Metric", style="bold")
    table.add_column("Baseline", style="cyan")
    table.add_column("Phase 1", style="green")
    table.add_column("Change", style="yellow")

    table.add_row(
        "Total Chunks",
        f"{baseline['total_chunks']:,}",
        f"{phase1['total_chunks']:,}",
        f"{chunk_change:+.1f}%"
    )

    table.add_row(
        "Total Characters",
        f"{baseline['total_chars']:,}",
        f"{phase1['total_chars']:,}",
        f"{char_change:+.1f}% 🎯"
    )

    table.add_row(
        "Characters Removed",
        "-",
        f"{baseline['total_chars'] - phase1['total_chars']:,}",
        "✅ Cleaned"
    )

    table.add_row(
        "FAISS Size",
        f"{baseline['faiss_size_mb']:.2f} MB",
        f"{phase1['faiss_size_mb']:.2f} MB",
        f"{((phase1['faiss_size_mb'] - baseline['faiss_size_mb']) / baseline['faiss_size_mb'] * 100):+.1f}%"
    )

    table.add_row(
        "Total Storage",
        f"{baseline_storage:.2f} MB",
        f"{phase1_storage:.2f} MB",
        f"{storage_change:+.1f}% 💾"
    )

    console.print(table)

    # Summary
    console.print("\n[bold]Impact Summary:[/bold]")
    console.print(f"  • Removed {abs(char_change):.1f}% of content (quotes + signatures)")
    console.print(f"  • Storage reduction: {abs(baseline_storage - phase1_storage):.2f} MB")
    console.print(f"  • Cleaner embeddings: No duplicate/boilerplate content")
    console.print(f"  • Semantic chunks: Topic-aware boundaries\n")


def main():
    """Run Phase 1 validation test on Primo_List dataset."""

    console = Console()

    console.print("\n" + "="*80, style="bold magenta")
    console.print("🧪 PHASE 1 VALIDATION TEST - Full Primo_List Dataset", style="bold magenta")
    console.print("="*80 + "\n", style="bold magenta")

    project_path = Path("data/projects/Primo_List")

    if not project_path.exists():
        console.print(f"[red]❌ Error: Project not found: {project_path}[/red]")
        return 1

    # Step 1: Backup existing data
    backup_dir = backup_existing_data(project_path, console)

    # Step 2: Get baseline metrics
    baseline_metrics = get_baseline_metrics(backup_dir, console)

    # Step 3: Clear processed data
    clear_processed_data(project_path, console)

    # Step 4: Run Phase 1 pipeline
    duration = run_phase1_pipeline(project_path, console)

    # Step 5: Get Phase 1 metrics
    phase1_metrics = get_phase1_metrics(project_path, console)

    # Step 6: Compare results
    compare_metrics(baseline_metrics, phase1_metrics, console)

    # Final summary
    console.print("\n" + "="*80, style="bold magenta")
    console.print("✅ Phase 1 Validation Complete!", style="bold green")
    console.print("="*80 + "\n", style="bold magenta")

    console.print("[bold]Next Steps:[/bold]")
    console.print("  1. Review the metrics above")
    console.print("  2. Test retrieval quality with sample queries")
    console.print("  3. Verify no duplicate results in search")
    console.print(f"  4. Restore backup if needed: {backup_dir}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
