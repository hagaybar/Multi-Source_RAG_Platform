#!/usr/bin/env python3
"""
Test quote deduplication on Primo_List emails.

Shows before/after comparison of reply emails with quoted text.
"""

import sys
import json
import random
from pathlib import Path
from typing import List, Dict
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.email.cleaning.quote_deduplicator import QuoteDeduplicator


def load_reply_emails(metadata_path: Path, n_samples: int = 5) -> List[Dict]:
    """Load emails that are likely replies (have Re: or quoted text)."""

    emails = []
    dedup = QuoteDeduplicator()

    with open(metadata_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                email = json.loads(line)
                text = email.get('text', '')
                subject = email.get('subject', '')

                # Only include emails that are likely replies
                if dedup.is_likely_reply(text, subject):
                    if 200 < len(text) < 2000:  # Reasonable length
                        emails.append(email)

    # Sample randomly
    if len(emails) > n_samples:
        emails = random.sample(emails, n_samples)

    return emails


def display_deduplication_results(email: Dict, cleaned: str, stats: dict, console: Console):
    """Display quote deduplication results."""

    # Email header
    subject = email.get('subject', 'No subject')[:80]
    sender = email.get('sender_name', 'Unknown')[:40]
    original_text = email.get('text', '')

    header = f"[bold cyan]From:[/bold cyan] {sender}\n"
    header += f"[bold cyan]Subject:[/bold cyan] {subject}"

    console.print(Panel(header, title="📧 Email", border_style="cyan"))

    # Stats table
    stats_table = Table(show_header=False, box=box.SIMPLE)
    stats_table.add_column("Metric", style="bold yellow")
    stats_table.add_column("Value", style="green")

    stats_table.add_row("Original Length", f"{stats['original_length']} chars")
    stats_table.add_row("Final Length", f"{stats['final_length']} chars")
    stats_table.add_row("Removed", f"{stats['removed_chars']} chars ({stats['reduction_ratio']:.1%})")
    stats_table.add_row("Lines Removed", str(stats['removed_lines']))
    stats_table.add_row("Method", stats['method'])
    if stats['quote_blocks_found'] > 0:
        stats_table.add_row("Quote Blocks", str(stats['quote_blocks_found']))

    console.print(Panel(stats_table, title="📊 Deduplication Stats", border_style="yellow"))

    # Before/After comparison
    console.print("\n[bold]BEFORE (with quoted text):[/bold]", style="red")
    console.print(Panel(
        original_text[:500] + ("..." if len(original_text) > 500 else ""),
        border_style="red",
        padding=(0, 1)
    ))

    console.print("\n[bold]AFTER (unique content only):[/bold]", style="green")
    console.print(Panel(
        cleaned[:500] + ("..." if len(cleaned) > 500 else ""),
        border_style="green",
        padding=(0, 1)
    ))

    console.print()


def main():
    """Test quote deduplication on Primo_List emails."""

    console = Console()

    console.print("\n" + "="*80, style="bold magenta")
    console.print("🎯 QUOTE DEDUPLICATION TEST - Primo_List Reply Emails", style="bold magenta")
    console.print("="*80 + "\n", style="bold magenta")

    # Paths
    project_path = Path("data/projects/Primo_List")
    metadata_path = project_path / "output" / "metadata" / "outlook_eml_metadata.jsonl"

    if not metadata_path.exists():
        console.print(f"❌ [red]Error:[/red] Metadata file not found: {metadata_path}")
        return 1

    # Load reply emails
    console.print("[cyan]Loading reply emails...[/cyan]")
    n_samples = 3
    emails = load_reply_emails(metadata_path, n_samples=n_samples)

    if not emails:
        console.print("[yellow]No reply emails found with quoted text.[/yellow]")
        console.print("This might mean:")
        console.print("  • Emails don't have quoted text (uncommon in mailing lists)")
        console.print("  • Quote markers are not standard (>, |)")
        return 0

    console.print(f"[green]✓ Found {len(emails)} reply emails with quoted text[/green]\n")

    # Initialize deduplicator
    dedup = QuoteDeduplicator(
        min_quote_length=50,
        similarity_threshold=0.85
    )

    # Process each email
    total_removed = 0
    total_original = 0

    for idx, email in enumerate(emails, 1):
        console.print(f"\n{'─'*80}", style="bold blue")
        console.print(f"Email {idx} of {len(emails)}", style="bold blue")
        console.print(f"{'─'*80}\n", style="bold blue")

        # Deduplicate
        text = email.get('text', '')
        cleaned, stats = dedup.deduplicate(text)

        # Track totals
        total_removed += stats['removed_chars']
        total_original += stats['original_length']

        # Display results
        display_deduplication_results(email, cleaned, stats, console)

    # Summary
    console.print("\n" + "="*80, style="bold magenta")
    console.print("📊 Summary", style="bold magenta")
    console.print("="*80 + "\n", style="bold magenta")

    overall_reduction = (
        total_removed / total_original if total_original > 0 else 0
    )

    summary = Table(show_header=False, box=box.SIMPLE)
    summary.add_column("Metric", style="bold")
    summary.add_column("Value", style="green")

    summary.add_row("Emails Processed", str(len(emails)))
    summary.add_row("Total Original", f"{total_original:,} chars")
    summary.add_row("Total Removed", f"{total_removed:,} chars")
    summary.add_row("Overall Reduction", f"{overall_reduction:.1%}")

    console.print(summary)

    console.print("\n[bold]Impact:[/bold]")
    console.print(f"  • Prevented indexing {total_removed:,} duplicate characters")
    console.print(f"  • Each email now contains only unique contribution")
    console.print(f"  • Storage reduction: ~{overall_reduction:.0%}")
    console.print(f"  • Better retrieval: No duplicate results from same thread\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
