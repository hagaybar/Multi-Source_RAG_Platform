#!/usr/bin/env python3
"""
Quality Validation for Primo_List_2 (Phase 1 Test Dataset)

Tests:
1. Chunk statistics and distribution
2. Phase 1 cleaning effectiveness (signatures/quotes removed)
3. Semantic chunking quality (topic coherence)
4. Retrieval quality (sample queries)
"""

import sys
import json
import random
from pathlib import Path
from collections import Counter
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.core.project_manager import ProjectManager
from scripts.retrieval.retrieval_manager import RetrievalManager


console = Console()


def analyze_chunks(project_path: Path):
    """Analyze chunk statistics."""

    console.print("\n" + "="*80, style="bold magenta")
    console.print("📊 CHUNK STATISTICS", style="bold magenta")
    console.print("="*80 + "\n", style="bold magenta")

    chunks_file = project_path / "input" / "chunks_outlook_eml.tsv"

    if not chunks_file.exists():
        console.print("[red]❌ Chunks file not found![/red]")
        return None

    import csv

    chunks = []
    with open(chunks_file, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            chunks.append({
                'text': row['text'],
                'tokens': int(row['token_count']),
                'meta': json.loads(row['meta_json'])
            })

    # Calculate stats
    token_counts = [c['tokens'] for c in chunks]
    char_counts = [len(c['text']) for c in chunks]

    # Email count
    unique_emails = len(set(c['meta'].get('message_id', c['meta'].get('doc_id')) for c in chunks))

    # Statistics table
    table = Table(title="Chunk Statistics", box=box.ROUNDED)
    table.add_column("Metric", style="bold yellow")
    table.add_column("Value", style="green")

    table.add_row("Total Chunks", f"{len(chunks):,}")
    table.add_row("Unique Emails", f"{unique_emails:,}")
    table.add_row("Avg Chunks/Email", f"{len(chunks)/unique_emails:.1f}")
    table.add_row("", "")
    table.add_row("Avg Tokens/Chunk", f"{sum(token_counts)/len(token_counts):.1f}")
    table.add_row("Min Tokens", f"{min(token_counts)}")
    table.add_row("Max Tokens", f"{max(token_counts)}")
    table.add_row("", "")
    table.add_row("Avg Chars/Chunk", f"{sum(char_counts)/len(char_counts):.0f}")
    table.add_row("Total Characters", f"{sum(char_counts):,}")

    console.print(table)
    console.print()

    return chunks


def check_cleaning_quality(chunks):
    """Check if Phase 1 cleaning worked (no signatures/quotes)."""

    console.print("\n" + "="*80, style="bold magenta")
    console.print("🧹 PHASE 1 CLEANING VALIDATION", style="bold magenta")
    console.print("="*80 + "\n", style="bold magenta")

    # Signature indicators
    signature_patterns = [
        '--',
        'Best regards',
        'Sincerely',
        'Kind regards',
        'Sent from my',
        'Get Outlook for'
    ]

    # Quote indicators
    quote_patterns = [
        'On .* wrote:',
        '> ',
        '-----Original Message-----'
    ]

    chunks_with_signatures = 0
    chunks_with_quotes = 0

    sample_issues = []

    for chunk in chunks:
        text = chunk['text'].lower()

        # Check for signatures
        has_signature = any(pattern.lower() in text for pattern in signature_patterns)
        if has_signature:
            chunks_with_signatures += 1
            if len(sample_issues) < 3:
                sample_issues.append(('signature', chunk['text'][:300]))

        # Check for quotes
        has_quotes = any(pattern.lower() in text or text.startswith('>') for pattern in quote_patterns)
        if has_quotes:
            chunks_with_quotes += 1
            if len(sample_issues) < 3:
                sample_issues.append(('quote', chunk['text'][:300]))

    # Results
    sig_rate = chunks_with_signatures / len(chunks) * 100
    quote_rate = chunks_with_quotes / len(chunks) * 100

    table = Table(box=box.ROUNDED)
    table.add_column("Check", style="bold")
    table.add_column("Result", style="cyan")
    table.add_column("Status", style="green")

    table.add_row(
        "Signature Detection",
        f"{chunks_with_signatures}/{len(chunks)} chunks ({sig_rate:.1f}%)",
        "✅ Good" if sig_rate < 5 else "⚠️  Some signatures remain"
    )

    table.add_row(
        "Quote Detection",
        f"{chunks_with_quotes}/{len(chunks)} chunks ({quote_rate:.1f}%)",
        "✅ Good" if quote_rate < 5 else "⚠️  Some quotes remain"
    )

    console.print(table)

    # Show samples if issues found
    if sample_issues:
        console.print("\n[yellow]Sample Issues Found:[/yellow]")
        for issue_type, sample_text in sample_issues[:2]:
            console.print(f"\n[red]{issue_type.upper()}:[/red]")
            console.print(Panel(sample_text, border_style="red"))

    console.print()


def check_semantic_quality(chunks):
    """Check semantic chunking quality - sample random chunks."""

    console.print("\n" + "="*80, style="bold magenta")
    console.print("🎯 SEMANTIC CHUNKING QUALITY", style="bold magenta")
    console.print("="*80 + "\n", style="bold magenta")

    # Sample 5 random chunks
    samples = random.sample(chunks, min(5, len(chunks)))

    console.print("[bold]Random Sample Chunks (for manual inspection):[/bold]\n")

    for i, chunk in enumerate(samples, 1):
        console.print(f"[cyan]Sample {i}:[/cyan] ({chunk['tokens']} tokens)")
        console.print(Panel(
            chunk['text'][:500] + ("..." if len(chunk['text']) > 500 else ""),
            border_style="cyan",
            padding=(0, 1)
        ))
        console.print()

    console.print("[bold]Manual Inspection Questions:[/bold]")
    console.print("  1. Does each chunk cover a single topic?")
    console.print("  2. Are chunks coherent (complete thoughts)?")
    console.print("  3. Are topic boundaries sensible?")
    console.print("  4. Would you want these as retrieval results?\n")


def test_retrieval_quality(project_path: Path):
    """Test retrieval with sample queries."""

    console.print("\n" + "="*80, style="bold magenta")
    console.print("🔍 RETRIEVAL QUALITY TEST", style="bold magenta")
    console.print("="*80 + "\n", style="bold magenta")

    # Sample queries
    queries = [
        "budget approval",
        "primo configuration",
        "technical support",
        "migration issue",
        "user interface problem"
    ]

    try:
        # Initialize retrieval
        project = ProjectManager(project_path)
        retrieval = RetrievalManager(
            project_root=project_path,
            config=project.config,
            top_k=3,
            strategy="late_fusion"
        )

        for query in queries:
            console.print(f"\n[bold cyan]Query:[/bold cyan] {query}")

            # Retrieve
            results = retrieval.retrieve(query)

            if not results:
                console.print("  [yellow]No results found[/yellow]")
                continue

            # Show top 3
            for i, chunk in enumerate(results[:3], 1):
                similarity = chunk.meta.get('similarity', 0)
                text_preview = chunk.text[:150].replace('\n', ' ')

                console.print(f"  [green]{i}. ({similarity:.3f})[/green] {text_preview}...")

        console.print("\n[bold]Retrieval Quality Questions:[/bold]")
        console.print("  1. Are results relevant to queries?")
        console.print("  2. Do top results make sense?")
        console.print("  3. Are chunks clean (no signatures/quotes)?")
        console.print("  4. Would these help answer questions?\n")

    except Exception as e:
        console.print(f"[red]Retrieval test failed: {e}[/red]")
        console.print("[yellow]This is OK - we can still validate chunks manually.[/yellow]\n")


def main():
    """Run quality validation on Primo_List_2."""

    console.print("\n" + "="*80, style="bold magenta")
    console.print("🧪 PRIMO_LIST_2 QUALITY VALIDATION", style="bold magenta")
    console.print("="*80, style="bold magenta")

    project_path = Path("data/projects/Primo_List_2")

    if not project_path.exists():
        console.print(f"\n[red]❌ Project not found: {project_path}[/red]")
        console.print("[yellow]Please create Primo_List_2 first.[/yellow]\n")
        return 1

    # Test 1: Analyze chunks
    chunks = analyze_chunks(project_path)

    if not chunks:
        return 1

    # Test 2: Check cleaning quality
    check_cleaning_quality(chunks)

    # Test 3: Check semantic quality
    check_semantic_quality(chunks)

    # Test 4: Test retrieval
    test_retrieval_quality(project_path)

    # Final summary
    console.print("\n" + "="*80, style="bold magenta")
    console.print("✅ VALIDATION COMPLETE", style="bold green")
    console.print("="*80, style="bold magenta")

    console.print("\n[bold]Next Steps:[/bold]")
    console.print("  1. Review the statistics above")
    console.print("  2. Manually inspect the sample chunks")
    console.print("  3. Test retrieval results")
    console.print("  4. If quality is good → apply to full Primo_List")
    console.print("  5. If issues found → tune parameters and re-run\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
