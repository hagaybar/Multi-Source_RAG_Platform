#!/usr/bin/env python3
"""Analyze new Phase 1 chunks before embedding."""
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

# Load new chunks
chunks_file = Path("data/projects/Primo_List/input/chunks_outlook_eml.tsv")
baseline_metadata = Path("data/projects/Primo_List/backup_pre_phase1/metadata/outlook_eml_metadata.jsonl")

# Analyze new chunks
new_chunks = []
with open(chunks_file, 'r') as f:
    lines = f.readlines()
    for line in lines[1:]:  # Skip header row
        if line.strip():
            parts = line.strip().split('\t')
            if len(parts) >= 5:
                chunk_id, doc_id, text, token_count, meta_json = parts[:5]
                try:
                    new_chunks.append({
                        'text': text,
                        'tokens': int(token_count),
                        'meta': json.loads(meta_json)
                    })
                except (ValueError, json.JSONDecodeError):
                    continue

# Analyze baseline
baseline_chunks = []
if baseline_metadata.exists():
    with open(baseline_metadata, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                baseline_chunks.append({
                    'text': data.get('text', ''),
                    'tokens': data.get('token_count', 0)
                })

# Calculate stats
def get_stats(chunks):
    if not chunks:
        return {}
    
    tokens = [c['tokens'] for c in chunks]
    chars = [len(c['text']) for c in chunks]
    
    return {
        'count': len(chunks),
        'total_tokens': sum(tokens),
        'total_chars': sum(chars),
        'avg_tokens': sum(tokens) / len(tokens),
        'min_tokens': min(tokens),
        'max_tokens': max(tokens),
        'avg_chars': sum(chars) / len(chars)
    }

baseline_stats = get_stats(baseline_chunks)
new_stats = get_stats(new_chunks)

# Display comparison
console.print("\n" + "="*80, style="bold magenta")
console.print("📊 CHUNK ANALYSIS - Phase 1 vs Baseline", style="bold magenta")
console.print("="*80 + "\n", style="bold magenta")

table = Table(title="Comparison", box=box.DOUBLE)
table.add_column("Metric", style="bold")
table.add_column("Baseline (Old)", style="cyan")
table.add_column("Phase 1 (New)", style="green")
table.add_column("Change", style="yellow")

# Chunk count
chunk_change = ((new_stats['count'] - baseline_stats['count']) / baseline_stats['count'] * 100) if baseline_stats.get('count') else 0
table.add_row(
    "Total Chunks",
    f"{baseline_stats.get('count', 0):,}",
    f"{new_stats['count']:,}",
    f"{chunk_change:+.1f}%"
)

# Total characters
char_change = ((new_stats['total_chars'] - baseline_stats['total_chars']) / baseline_stats['total_chars'] * 100) if baseline_stats.get('total_chars') else 0
table.add_row(
    "Total Characters",
    f"{baseline_stats.get('total_chars', 0):,}",
    f"{new_stats['total_chars']:,}",
    f"{char_change:+.1f}% 🎯"
)

# Average chunk size
table.add_row(
    "Avg Chunk Size",
    f"{baseline_stats.get('avg_chars', 0):.0f} chars",
    f"{new_stats['avg_chars']:.0f} chars",
    f"{((new_stats['avg_chars'] - baseline_stats.get('avg_chars', 0)) / baseline_stats.get('avg_chars', 1) * 100):+.1f}%"
)

# Token distribution
table.add_row(
    "Avg Tokens/Chunk",
    f"{baseline_stats.get('avg_tokens', 0):.0f}",
    f"{new_stats['avg_tokens']:.0f}",
    "-"
)

table.add_row(
    "Token Range",
    f"{baseline_stats.get('min_tokens', 0)}-{baseline_stats.get('max_tokens', 0)}",
    f"{new_stats['min_tokens']}-{new_stats['max_tokens']}",
    "-"
)

console.print(table)

# Sample chunks
console.print("\n[bold]Sample New Chunks (First 3):[/bold]\n")
for i, chunk in enumerate(new_chunks[:3], 1):
    console.print(f"[cyan]Chunk {i}:[/cyan] ({chunk['tokens']} tokens)")
    console.print(f"  {chunk['text'][:200]}...\n")

console.print("[bold]Phase 1 Impact:[/bold]")
chars_removed = baseline_stats.get('total_chars', 0) - new_stats['total_chars']
console.print(f"  • Removed {chars_removed:,} chars via cleaning ({abs(char_change):.1f}% reduction)")
console.print(f"  • Chunk count changed by {chunk_change:+.1f}%")
console.print(f"  • Semantic chunking: {'✅ Active' if abs(chunk_change) > 5 else '⚠️  May not be active'}")

if abs(char_change) > 10:
    console.print(f"\n✅ [green]Significant content reduction! Phase 1 is working.[/green]")
elif abs(char_change) > 2:
    console.print(f"\n✅ [green]Moderate content reduction. Phase 1 cleaning applied.[/green]")
else:
    console.print(f"\n⚠️  [yellow]Low reduction - check if cleaning is working as expected.[/yellow]")

console.print()
