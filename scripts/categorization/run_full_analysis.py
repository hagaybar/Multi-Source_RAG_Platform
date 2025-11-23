#!/usr/bin/env python3
"""
Run Complete Phase 1-3 Email Categorization Analysis

Executes the full email categorization pipeline:
- Phase 1-2: Category discovery (BERTopic + K-means)
- Phase 3.1: Thread-based grouping
- Phase 3.2: Temporal topic detection
- Phase 3.3: Hierarchical categories

Usage:
    python scripts/categorization/run_full_analysis.py \
        --project data/projects/YourProject \
        --n-categories 7 \
        --output data/categories/full_analysis.json

Requirements:
    - Project must have embeddings completed
    - Minimum 100 emails for meaningful results
    - Recommended: 300+ emails, 3+ months for full Phase 3
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.core.project_manager import ProjectManager
from scripts.categorization.bertopic_discovery import BERTopicCategoryDiscovery
from scripts.categorization.thread_analyzer import EmailThreadAnalyzer
from scripts.categorization.temporal_analyzer import TemporalTopicAnalyzer
from scripts.chunking.models import Chunk


def load_email_chunks(project: ProjectManager) -> list[Chunk]:
    """Load email chunks from metadata JSONL."""
    print("\n" + "="*60)
    print("Loading Email Chunks")
    print("="*60)

    metadata_path = project.metadata_dir / "outlook_eml_metadata.jsonl"

    if not metadata_path.exists():
        print(f"❌ Metadata file not found: {metadata_path}")
        print("   Make sure embeddings have been completed.")
        return []

    chunks = []
    with open(metadata_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                chunk = Chunk(
                    id=data.get('id', ''),
                    doc_id=data.get('source_filepath', ''),
                    text=data.get('text', ''),
                    token_count=data.get('token_count', 0),
                    meta=data
                )
                chunks.append(chunk)

    print(f"✓ Loaded {len(chunks)} email chunks")
    return chunks


def run_phase_1_2_discovery(project: ProjectManager, n_categories: int,
                            output_path: Path) -> dict:
    """Run Phase 1-2: Category Discovery."""
    print("\n" + "="*80)
    print("PHASE 1-2: CATEGORY DISCOVERY (BERTopic + K-means)")
    print("="*80)

    discovery = BERTopicCategoryDiscovery(project)
    discovery.run_bertopic(
        use_kmeans=True,
        n_clusters=n_categories,
        output_path=output_path,
        interactive=False
    )

    # Return discovered categories
    return discovery.category_mapping


def run_phase_3_1_threads(chunks: list[Chunk]) -> dict:
    """Run Phase 3.1: Thread Analysis."""
    print("\n" + "="*80)
    print("PHASE 3.1: THREAD-BASED GROUPING")
    print("="*80)

    analyzer = EmailThreadAnalyzer()

    print("\nGrouping emails into threads...")
    threads = analyzer.group_by_thread(chunks)

    print(f"✓ Found {len(threads)} threads")

    # Print statistics
    analyzer.print_thread_stats()

    # Analyze thread categories
    print("\nAnalyzing thread category coherence...")
    analysis = analyzer.analyze_thread_categories()

    # Print coherence
    coherence = analyzer.get_thread_coherence_score()
    print(f"\nOverall thread coherence: {coherence*100:.1f}%")

    return {
        'total_threads': len(threads),
        'thread_stats': analyzer.thread_stats,
        'coherence_score': coherence,
        'category_analysis': analysis
    }


def run_phase_3_2_temporal(chunks: list[Chunk]) -> dict:
    """Run Phase 3.2: Temporal Analysis."""
    print("\n" + "="*80)
    print("PHASE 3.2: TEMPORAL TOPIC DETECTION")
    print("="*80)

    analyzer = TemporalTopicAnalyzer(time_unit='month')

    print("\nAnalyzing temporal evolution...")
    temporal_topics = analyzer.analyze_temporal_evolution(chunks)

    print(f"✓ Analyzed {len(temporal_topics)} categories")

    # Print analysis
    analyzer.print_temporal_analysis()

    # Get trending and time-bound topics
    trending = analyzer.get_trending_topics()
    time_bound = analyzer.get_time_bound_topics()

    return {
        'temporal_topics': temporal_topics,
        'trending_topics': trending,
        'time_bound_topics': time_bound,
        'has_meaningful_data': len(temporal_topics) > 0 and \
                               any(data['active_periods'] > 1 for data in temporal_topics.values())
    }


def run_phase_3_3_hierarchical(project: ProjectManager, output_path: Path) -> dict:
    """Run Phase 3.3: Hierarchical Categories."""
    print("\n" + "="*80)
    print("PHASE 3.3: HIERARCHICAL CATEGORIES")
    print("="*80)

    discovery = BERTopicCategoryDiscovery(project)

    print("\nCreating hierarchical category structure...")
    hierarchy = discovery.create_hierarchical_categories(
        max_fine_topics=15,
        n_broad_categories=5,
        output_path=output_path
    )

    if not hierarchy:
        print("⚠️  Hierarchical analysis skipped (insufficient data)")
        return {'skipped': True, 'reason': 'Dataset too small (<300 emails)'}

    return hierarchy


def generate_summary_report(results: dict, output_path: Path):
    """Generate comprehensive summary report."""
    print("\n" + "="*80)
    print("GENERATING SUMMARY REPORT")
    print("="*80)

    report = {
        'analysis_date': datetime.now().isoformat(),
        'project': results.get('project_name'),
        'dataset_info': results.get('dataset_info'),
        'phase_1_2': {
            'categories': results.get('categories'),
            'category_count': len(results.get('categories', {}))
        },
        'phase_3_1': results.get('thread_analysis'),
        'phase_3_2': results.get('temporal_analysis'),
        'phase_3_3': results.get('hierarchical_analysis'),
        'recommendations': []
    }

    # Add recommendations based on results
    dataset_size = results.get('dataset_info', {}).get('total_emails', 0)

    if dataset_size < 100:
        report['recommendations'].append(
            "⚠️  Small dataset (<100 emails). Results may be limited. "
            "Consider using 300+ emails for robust analysis."
        )

    if not results.get('temporal_analysis', {}).get('has_meaningful_data'):
        report['recommendations'].append(
            "⚠️  Insufficient temporal data. For temporal analysis, "
            "use 2-3+ months of emails."
        )

    if results.get('hierarchical_analysis', {}).get('skipped'):
        report['recommendations'].append(
            "⚠️  Hierarchical analysis skipped (dataset too small). "
            "Use 300+ emails for hierarchical categories."
        )

    if results.get('thread_analysis', {}).get('coherence_score', 0) < 0.7:
        report['recommendations'].append(
            "💡 Thread coherence is low (<70%). Consider running "
            "thread recategorization for better consistency."
        )

    # Save report
    report_path = output_path.parent / f"{output_path.stem}_summary.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"✓ Summary report saved to: {report_path}")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Run complete Phase 1-3 email categorization analysis"
    )
    parser.add_argument("--project", type=str, required=True,
                       help="Path to project directory")
    parser.add_argument("--n-categories", type=int, default=7,
                       help="Number of categories to discover (default: 7)")
    parser.add_argument("--output", type=str,
                       default="data/categories/full_analysis.json",
                       help="Output path for results")
    parser.add_argument("--skip-hierarchical", action="store_true",
                       help="Skip hierarchical analysis (for small datasets)")
    parser.add_argument("--skip-temporal", action="store_true",
                       help="Skip temporal analysis (for short time spans)")

    args = parser.parse_args()

    print("="*80)
    print("  COMPLETE EMAIL CATEGORIZATION ANALYSIS (PHASES 1-3)")
    print("="*80)
    print(f"\nProject: {args.project}")
    print(f"Target categories: {args.n_categories}")
    print(f"Output: {args.output}")

    # Initialize
    project = ProjectManager(Path(args.project))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        'project_name': project.config.get('project.name', 'Unknown')
    }

    # Load chunks
    chunks = load_email_chunks(project)

    if not chunks:
        print("\n❌ No email chunks found. Cannot proceed.")
        print("   Make sure you've run: ingest → chunk → embed")
        return 1

    results['dataset_info'] = {
        'total_chunks': len(chunks),
        'total_emails': len(set(c.meta.get('subject', '') for c in chunks)),
        'date_range': {
            'first': min(c.meta.get('date', '') for c in chunks if c.meta.get('date')),
            'last': max(c.meta.get('date', '') for c in chunks if c.meta.get('date'))
        }
    }

    print(f"\nDataset: {results['dataset_info']['total_chunks']} chunks, "
          f"~{results['dataset_info']['total_emails']} unique subjects")

    # Phase 1-2: Category Discovery
    categories = run_phase_1_2_discovery(project, args.n_categories, output_path)
    results['categories'] = categories

    # Phase 3.1: Thread Analysis
    thread_results = run_phase_3_1_threads(chunks)
    results['thread_analysis'] = thread_results

    # Phase 3.2: Temporal Analysis (unless skipped)
    if not args.skip_temporal:
        temporal_results = run_phase_3_2_temporal(chunks)
        results['temporal_analysis'] = temporal_results
    else:
        print("\n⏭️  Skipping temporal analysis (--skip-temporal)")
        results['temporal_analysis'] = {'skipped': True}

    # Phase 3.3: Hierarchical Categories (unless skipped or too small)
    if not args.skip_hierarchical and len(chunks) >= 300:
        hierarchical_output = output_path.parent / f"{output_path.stem}_hierarchical.json"
        hierarchical_results = run_phase_3_3_hierarchical(project, hierarchical_output)
        results['hierarchical_analysis'] = hierarchical_results
    else:
        reason = "User requested skip" if args.skip_hierarchical else "Dataset too small"
        print(f"\n⏭️  Skipping hierarchical analysis ({reason})")
        results['hierarchical_analysis'] = {'skipped': True, 'reason': reason}

    # Generate summary report
    summary = generate_summary_report(results, output_path)

    # Print final summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\n✅ Phase 1-2: {len(categories)} categories discovered")
    print(f"✅ Phase 3.1: {thread_results['total_threads']} threads detected "
          f"({thread_results['coherence_score']*100:.1f}% coherence)")

    if not args.skip_temporal:
        temporal_meaningful = results['temporal_analysis'].get('has_meaningful_data', False)
        status = "✅" if temporal_meaningful else "⚠️"
        print(f"{status} Phase 3.2: Temporal analysis {'complete' if temporal_meaningful else 'limited (short timespan)'}")

    if not args.skip_hierarchical:
        hierarchical_skipped = results['hierarchical_analysis'].get('skipped', False)
        status = "✅" if not hierarchical_skipped else "⏭️"
        print(f"{status} Phase 3.3: Hierarchical analysis {'complete' if not hierarchical_skipped else 'skipped'}")

    print(f"\n📊 Results saved to: {output_path}")
    print(f"📄 Summary report: {output_path.parent / f'{output_path.stem}_summary.json'}")

    if summary.get('recommendations'):
        print("\n💡 Recommendations:")
        for rec in summary['recommendations']:
            print(f"   {rec}")

    print("\n" + "="*80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
