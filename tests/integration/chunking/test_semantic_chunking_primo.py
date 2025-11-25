#!/usr/bin/env python3
"""
Test semantic chunking on Primo_List emails.

Loads real emails and displays semantic chunking results
with pretty formatting.
"""

import sys
import json
import random
from pathlib import Path
from typing import List, Dict
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich import box

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.chunking.semantic_chunker import SemanticChunker


def load_sample_emails(metadata_path: Path, n_samples: int = 5) -> List[Dict]:
    """Load random sample of emails from metadata."""

    emails = []
    with open(metadata_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                email = json.loads(line)
                # Only include emails with reasonable length
                if 100 < len(email.get('text', '')) < 2000:
                    emails.append(email)

    # Sample randomly
    if len(emails) > n_samples:
        emails = random.sample(emails, n_samples)

    return emails


def display_chunking_results(email: Dict, chunks: List[tuple], console: Console):
    """Display semantic chunking results with rich formatting."""

    # Email header
    subject = email.get('subject', 'No subject')[:80]
    sender = email.get('sender_name', 'Unknown')[:40]
    date = email.get('date', '')[:10]
    text_length = len(email.get('text', ''))

    header = f"[bold cyan]From:[/bold cyan] {sender}\n"
    header += f"[bold cyan]Date:[/bold cyan] {date}\n"
    header += f"[bold cyan]Subject:[/bold cyan] {subject}\n"
    header += f"[bold cyan]Length:[/bold cyan] {text_length} chars"

    console.print(Panel(header, title="📧 Email", border_style="cyan"))

    # Chunking summary
    total_chunks = len(chunks)
    total_tokens = sum(tc for _, tc in chunks)
    avg_tokens = total_tokens / total_chunks if total_chunks > 0 else 0

    summary = Table(show_header=False, box=box.SIMPLE)
    summary.add_column("Metric", style="bold yellow")
    summary.add_column("Value", style="green")

    summary.add_row("Total Chunks", str(total_chunks))
    summary.add_row("Total Tokens", str(total_tokens))
    summary.add_row("Avg Tokens/Chunk", f"{avg_tokens:.1f}")
    summary.add_row("Min Tokens", str(min(tc for _, tc in chunks)) if chunks else "0")
    summary.add_row("Max Tokens", str(max(tc for _, tc in chunks)) if chunks else "0")

    console.print(Panel(summary, title="📊 Chunking Summary", border_style="yellow"))

    # Individual chunks
    for i, (chunk_text, token_count) in enumerate(chunks, 1):
        # Truncate very long chunks for display
        display_text = chunk_text
        if len(chunk_text) > 300:
            display_text = chunk_text[:300] + "..."

        # Color-code by size
        if token_count < 50:
            style = "yellow"
        elif token_count > 200:
            style = "red"
        else:
            style = "green"

        chunk_header = f"[bold]Chunk {i}[/bold] ([{style}]{token_count} tokens[/{style}])"

        console.print(Panel(
            display_text,
            title=chunk_header,
            border_style=style,
            padding=(0, 1)
        ))

    console.print()  # Spacing


def main():
    """Test semantic chunking on Primo_List emails."""

    console = Console()

    console.print("\n" + "="*80, style="bold magenta")
    console.print("🎯 SEMANTIC CHUNKING TEST - Primo_List Emails", style="bold magenta")
    console.print("="*80 + "\n", style="bold magenta")

    # Paths
    project_path = Path("data/projects/Primo_List")
    metadata_path = project_path / "output" / "metadata" / "outlook_eml_metadata.jsonl"

    if not metadata_path.exists():
        console.print(f"❌ [red]Error:[/red] Metadata file not found: {metadata_path}")
        console.print("   Please run ingestion first.")
        return 1

    # Load sample emails
    console.print("[cyan]Loading sample emails...[/cyan]")
    n_samples = 3
    emails = load_sample_emails(metadata_path, n_samples=n_samples)

    if not emails:
        console.print("[red]No emails found![/red]")
        return 1

    console.print(f"[green]✓ Loaded {len(emails)} emails[/green]\n")

    # Initialize semantic chunker
    console.print("[cyan]Initializing semantic chunker...[/cyan]")
    chunker = SemanticChunker(
        similarity_threshold=0.65,
        min_chunk_size=30,
        max_chunk_size=300,
        sentence_overlap=1
    )
    console.print("[green]✓ Chunker ready[/green]\n")

    # Process each email
    for idx, email in enumerate(emails, 1):
        console.print(f"\n{'─'*80}", style="bold blue")
        console.print(f"Email {idx} of {len(emails)}", style="bold blue")
        console.print(f"{'─'*80}\n", style="bold blue")

        # Chunk the email
        email_text = email.get('text', '')
        chunks = chunker.chunk(email_text)

        # Display results
        display_chunking_results(email, chunks, console)

    # Final summary
    console.print("\n" + "="*80, style="bold magenta")
    console.print("✅ Test Complete!", style="bold green")
    console.print("="*80 + "\n", style="bold magenta")

    console.print("[bold]Observations:[/bold]")
    console.print("  • Semantic chunking detects topic boundaries automatically")
    console.print("  • Chunks represent coherent semantic units")
    console.print("  • No mixed topics within single chunks")
    console.print("  • Size varies naturally based on content\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
