#!/usr/bin/env python3
"""
Test signature detection on Primo_List emails.

Shows before/after comparison with signatures removed.
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

from scripts.email.cleaning.signature_detector import SignatureDetector


def load_sample_emails(metadata_path: Path, n_samples: int = 5) -> List[Dict]:
    """Load random sample of emails from metadata."""

    emails = []
    with open(metadata_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                email = json.loads(line)
                # Include emails with reasonable length
                text = email.get('text', '')
                if 200 < len(text) < 3000:
                    emails.append(email)

    # Sample randomly
    if len(emails) > n_samples:
        emails = random.sample(emails, n_samples)

    return emails


def display_signature_results(
    email: Dict,
    content: str,
    signature: str,
    stats: dict,
    console: Console
):
    """Display signature detection results."""

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
    stats_table.add_row("Cleaned Length", f"{stats['cleaned_length']} chars")
    stats_table.add_row("Signature Found", "✅ Yes" if stats['had_signature'] else "❌ No")

    if stats['had_signature']:
        stats_table.add_row("Removed", f"{stats['removed_chars']} chars ({stats['reduction_ratio']:.1%})")
        stats_table.add_row("Lines Removed", str(stats['removed_lines']))

    console.print(Panel(stats_table, title="📊 Detection Stats", border_style="yellow"))

    # Before/After comparison
    console.print("\n[bold]ORIGINAL (with signature):[/bold]", style="red")
    console.print(Panel(
        original_text[:600] + ("..." if len(original_text) > 600 else ""),
        border_style="red",
        padding=(0, 1)
    ))

    console.print("\n[bold]CONTENT (signature removed):[/bold]", style="green")
    console.print(Panel(
        content[:600] + ("..." if len(content) > 600 else ""),
        border_style="green",
        padding=(0, 1)
    ))

    if signature:
        console.print("\n[bold]SIGNATURE (detected):[/bold]", style="magenta")
        console.print(Panel(
            signature[:400] + ("..." if len(signature) > 400 else ""),
            border_style="magenta",
            padding=(0, 1)
        ))

    console.print()


def main():
    """Test signature detection on Primo_List emails."""

    console = Console()

    console.print("\n" + "="*80, style="bold magenta")
    console.print("🎯 SIGNATURE DETECTION TEST - Primo_List Emails", style="bold magenta")
    console.print("="*80 + "\n", style="bold magenta")

    # Paths
    project_path = Path("data/projects/Primo_List")
    metadata_path = project_path / "output" / "metadata" / "outlook_eml_metadata.jsonl"

    if not metadata_path.exists():
        console.print(f"❌ [red]Error:[/red] Metadata file not found: {metadata_path}")
        return 1

    # Load sample emails
    console.print("[cyan]Loading sample emails...[/cyan]")
    n_samples = 5
    emails = load_sample_emails(metadata_path, n_samples=n_samples)

    if not emails:
        console.print("[red]No suitable emails found![/red]")
        return 1

    console.print(f"[green]✓ Loaded {len(emails)} emails[/green]\n")

    # Initialize signature detector
    detector = SignatureDetector(
        min_signature_length=10,
        max_signature_length=500,
        confidence_threshold=0.65
    )

    # Process each email
    total_removed = 0
    total_original = 0
    emails_with_signatures = 0

    for idx, email in enumerate(emails, 1):
        console.print(f"\n{'─'*80}", style="bold blue")
        console.print(f"Email {idx} of {len(emails)}", style="bold blue")
        console.print(f"{'─'*80}\n", style="bold blue")

        # Detect and remove signature
        text = email.get('text', '')
        content, signature = detector.detect_signature(text)
        stats = detector.get_stats(text, content, signature)

        # Track totals
        total_removed += stats['removed_chars']
        total_original += stats['original_length']
        if stats['had_signature']:
            emails_with_signatures += 1

        # Display results
        display_signature_results(email, content, signature, stats, console)

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
    summary.add_row("Emails with Signatures", f"{emails_with_signatures} ({emails_with_signatures/len(emails):.1%})")
    summary.add_row("Total Original", f"{total_original:,} chars")
    summary.add_row("Total Removed", f"{total_removed:,} chars")
    summary.add_row("Overall Reduction", f"{overall_reduction:.1%}")

    console.print(summary)

    console.print("\n[bold]Impact:[/bold]")
    console.print(f"  • Prevented indexing {total_removed:,} signature characters")
    console.print(f"  • Cleaner embeddings without signature noise")
    console.print(f"  • Storage reduction: ~{overall_reduction:.0%}")
    console.print(f"  • Better retrieval: No signature-dominated results\n")

    # Next steps recommendation
    console.print("[bold yellow]Next Steps:[/bold yellow]")
    console.print("  1. Combine with quote deduplication for maximum cleaning")
    console.print("  2. Integrate into email ingestion pipeline")
    console.print("  3. Test on full Primo dataset")
    console.print("  4. Consider ML-based detection for edge cases\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
