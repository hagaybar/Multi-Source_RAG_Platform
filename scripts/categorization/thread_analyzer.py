#!/usr/bin/env python3
"""
Email Thread Analyzer - Thread-Based Grouping for Categorization

Groups related emails into conversation threads and categorizes them as a unit
for improved category coherence and quality.
"""

import re
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from datetime import datetime

from scripts.chunking.models import Chunk


class EmailThreadAnalyzer:
    """Analyzes email threads for better categorization.

    Features:
    - Normalize subject lines (remove Re:, Fwd:, tags)
    - Group emails into threads by subject
    - Detect conversation patterns
    - Categorize threads as a unit
    - Propagate categories to all emails in thread
    """

    def __init__(self):
        """Initialize thread analyzer."""
        self.threads = {}
        self.thread_stats = {}

    @staticmethod
    def normalize_subject(subject: str) -> str:
        """Normalize subject for thread matching.

        Removes:
        - [Primo], [ALMA-L], etc. mailing list tags
        - Re:, Fwd:, Fw: prefixes
        - Extra whitespace

        Args:
            subject: Original email subject

        Returns:
            Normalized subject string (lowercase, cleaned)
        """
        if not subject:
            return ""

        # Remove mailing list tags [TAG]
        normalized = re.sub(r'\[.*?\]', '', subject)

        # Remove Re:, Fwd:, Fw:, etc. (case insensitive, multiple times)
        normalized = re.sub(r'^(re|fwd?|fw):\s*', '', normalized, flags=re.IGNORECASE)
        # Handle multiple Re: Re: Re:
        while re.match(r'^(re|fwd?|fw):\s*', normalized, flags=re.IGNORECASE):
            normalized = re.sub(r'^(re|fwd?|fw):\s*', '', normalized, flags=re.IGNORECASE)

        # Normalize whitespace
        normalized = ' '.join(normalized.split())

        return normalized.lower().strip()

    def group_by_thread(self, emails: List[Chunk]) -> Dict[str, List[Chunk]]:
        """Group emails into conversation threads.

        Args:
            emails: List of email chunks

        Returns:
            Dict mapping thread_key → list of emails in that thread
        """
        threads = defaultdict(list)

        for email in emails:
            subject = email.meta.get('subject', '')
            thread_key = self.normalize_subject(subject)

            if not thread_key:
                # Create unique key for emails without subject
                thread_key = f"no_subject_{email.id[:8]}"

            threads[thread_key].append(email)

        # Sort emails within each thread by date
        for thread_key in threads:
            threads[thread_key].sort(
                key=lambda e: e.meta.get('date', ''),
                reverse=False  # Oldest first
            )

        self.threads = dict(threads)
        self._compute_thread_stats()

        return self.threads

    def _compute_thread_stats(self):
        """Compute statistics about threads."""
        if not self.threads:
            return

        thread_lengths = [len(emails) for emails in self.threads.values()]

        self.thread_stats = {
            'total_threads': len(self.threads),
            'total_emails': sum(thread_lengths),
            'min_length': min(thread_lengths),
            'max_length': max(thread_lengths),
            'avg_length': np.mean(thread_lengths),
            'median_length': np.median(thread_lengths),
            'singleton_threads': sum(1 for length in thread_lengths if length == 1),
            'multi_email_threads': sum(1 for length in thread_lengths if length > 1),
        }

    def get_thread_stats(self) -> Dict:
        """Get thread statistics.

        Returns:
            Dict with thread statistics
        """
        return self.thread_stats

    def print_thread_stats(self):
        """Print thread statistics."""
        if not self.thread_stats:
            print("No thread statistics available. Run group_by_thread() first.")
            return

        stats = self.thread_stats
        print("\n" + "="*60)
        print("THREAD ANALYSIS STATISTICS")
        print("="*60)
        print(f"Total threads: {stats['total_threads']}")
        print(f"Total emails: {stats['total_emails']}")
        print(f"Singleton threads: {stats['singleton_threads']} ({stats['singleton_threads']/stats['total_threads']*100:.1f}%)")
        print(f"Multi-email threads: {stats['multi_email_threads']} ({stats['multi_email_threads']/stats['total_threads']*100:.1f}%)")
        print(f"\nThread length:")
        print(f"  Min: {stats['min_length']}")
        print(f"  Max: {stats['max_length']}")
        print(f"  Avg: {stats['avg_length']:.1f}")
        print(f"  Median: {stats['median_length']:.0f}")
        print("="*60)

    def categorize_thread(self, thread: List[Chunk],
                         categorizer) -> Tuple[str, float]:
        """Categorize entire thread (aggregate signals from all emails).

        Args:
            thread: List of emails in the thread
            categorizer: Categorization object with categorize() method

        Returns:
            Tuple of (category, confidence)
        """
        if not thread:
            return ("Unknown", 0.0)

        if len(thread) == 1:
            # Single email thread - categorize normally
            return categorizer.categorize(thread[0])

        # Multi-email thread - combine content
        combined_text = self._combine_thread_content(thread)

        # Create synthetic "thread email" for categorization
        thread_email = self._create_thread_chunk(thread, combined_text)

        # Categorize
        category, confidence = categorizer.categorize(thread_email)

        return category, confidence

    def _combine_thread_content(self, thread: List[Chunk]) -> str:
        """Combine content from all emails in thread.

        Args:
            thread: List of emails in the thread

        Returns:
            Combined text content
        """
        # Combine subjects and bodies
        subjects = [email.meta.get('subject', '') for email in thread]
        bodies = [email.text for email in thread]

        # Use first subject as primary
        primary_subject = subjects[0] if subjects else ""

        # Combine all bodies
        combined_body = "\n\n".join(bodies)

        # Create combined text
        combined = f"{primary_subject}\n\n{combined_body}"

        return combined

    def _create_thread_chunk(self, thread: List[Chunk], combined_text: str) -> Chunk:
        """Create a synthetic chunk representing the entire thread.

        Args:
            thread: List of emails in the thread
            combined_text: Combined text from thread

        Returns:
            Chunk object representing the thread
        """
        # Use first email as template
        first_email = thread[0]

        # Create thread chunk
        thread_chunk = Chunk(
            id=f"thread_{first_email.id}",
            doc_id=first_email.doc_id,
            text=combined_text,
            token_count=len(combined_text.split()),  # Approximate
            meta={
                'doc_type': first_email.meta.get('doc_type', 'outlook_eml'),
                'subject': first_email.meta.get('subject', ''),
                'sender': first_email.meta.get('sender', ''),
                'date': first_email.meta.get('date', ''),
                'thread_size': len(thread),
                'is_thread_chunk': True,
            }
        )

        # If first email has embedding, use it (approximate)
        if hasattr(first_email, 'embedding') and first_email.embedding is not None:
            thread_chunk.embedding = first_email.embedding

        return thread_chunk

    def propagate_category_to_thread(self, thread: List[Chunk],
                                    category: str, confidence: float,
                                    source: str = 'thread_analysis'):
        """Apply category to all emails in thread.

        Args:
            thread: List of emails in the thread
            category: Category to apply
            confidence: Confidence score
            source: Source of categorization
        """
        for email in thread:
            email.meta['category'] = category
            email.meta['category_confidence'] = confidence
            email.meta['category_source'] = source
            email.meta['thread_size'] = len(thread)

    def detect_mixed_categories(self, thread: List[Chunk]) -> bool:
        """Check if thread has emails with different categories.

        Args:
            thread: List of emails in the thread

        Returns:
            True if mixed categories, False otherwise
        """
        categories = [email.meta.get('category') for email in thread
                     if 'category' in email.meta]

        if not categories:
            return False

        return len(set(categories)) > 1

    def recategorize_mixed_threads(self, categorizer, min_thread_size: int = 3):
        """Re-categorize threads that have mixed categories.

        Args:
            categorizer: Categorization object with categorize() method
            min_thread_size: Minimum thread size to recategorize

        Returns:
            Number of threads recategorized
        """
        if not self.threads:
            print("No threads available. Run group_by_thread() first.")
            return 0

        recategorized_count = 0

        for thread_key, thread_emails in self.threads.items():
            # Only recategorize multi-email threads
            if len(thread_emails) < min_thread_size:
                continue

            # Check if thread has mixed categories
            if self.detect_mixed_categories(thread_emails):
                # Re-categorize thread as whole
                category, confidence = self.categorize_thread(thread_emails, categorizer)

                # Propagate to all emails in thread
                self.propagate_category_to_thread(
                    thread_emails,
                    category,
                    confidence,
                    source='thread_recategorization'
                )

                recategorized_count += 1

        return recategorized_count

    def get_thread_coherence_score(self) -> float:
        """Calculate thread coherence score (% of threads with single category).

        Returns:
            Coherence score (0.0 to 1.0)
        """
        if not self.threads:
            return 0.0

        coherent_threads = 0

        for thread_emails in self.threads.values():
            if len(thread_emails) == 1:
                # Singleton threads are always coherent
                coherent_threads += 1
            elif not self.detect_mixed_categories(thread_emails):
                # Multi-email thread with single category
                coherent_threads += 1

        return coherent_threads / len(self.threads)

    def analyze_thread_categories(self) -> Dict:
        """Analyze category distribution within threads.

        Returns:
            Dict with analysis results
        """
        if not self.threads:
            return {}

        thread_category_stats = defaultdict(lambda: {
            'total_threads': 0,
            'coherent_threads': 0,
            'mixed_threads': 0,
            'avg_thread_size': 0,
        })

        for thread_emails in self.threads.values():
            # Get primary category (most common in thread)
            categories = [e.meta.get('category', 'Unknown') for e in thread_emails]
            primary_category = max(set(categories), key=categories.count)

            stats = thread_category_stats[primary_category]
            stats['total_threads'] += 1
            stats['avg_thread_size'] += len(thread_emails)

            if self.detect_mixed_categories(thread_emails):
                stats['mixed_threads'] += 1
            else:
                stats['coherent_threads'] += 1

        # Calculate averages
        for category, stats in thread_category_stats.items():
            if stats['total_threads'] > 0:
                stats['avg_thread_size'] /= stats['total_threads']
                stats['coherence_rate'] = stats['coherent_threads'] / stats['total_threads']

        return dict(thread_category_stats)

    def print_thread_category_analysis(self):
        """Print thread category analysis."""
        analysis = self.analyze_thread_categories()

        if not analysis:
            print("No thread category analysis available.")
            return

        print("\n" + "="*60)
        print("THREAD CATEGORY ANALYSIS")
        print("="*60)

        for category, stats in sorted(analysis.items(),
                                     key=lambda x: x[1]['total_threads'],
                                     reverse=True):
            print(f"\n{category}:")
            print(f"  Total threads: {stats['total_threads']}")
            print(f"  Coherent threads: {stats['coherent_threads']} ({stats['coherence_rate']*100:.1f}%)")
            print(f"  Mixed threads: {stats['mixed_threads']}")
            print(f"  Avg thread size: {stats['avg_thread_size']:.1f} emails")

        overall_coherence = self.get_thread_coherence_score()
        print(f"\n{'='*60}")
        print(f"Overall thread coherence: {overall_coherence*100:.1f}%")
        print("="*60)


def main():
    """Test thread analyzer with sample data."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from scripts.core.project_manager import ProjectManager

    # Test on Primo_List_2
    project = ProjectManager(Path("data/projects/Primo_List_2"))

    print("="*60)
    print("  EMAIL THREAD ANALYZER TEST")
    print("="*60)

    # Load email chunks from metadata JSONL (more robust than TSV)
    print("\nLoading email chunks...")
    chunks = []
    import json

    metadata_path = project.metadata_dir / "outlook_eml_metadata.jsonl"

    if not metadata_path.exists():
        print(f"❌ Metadata file not found: {metadata_path}")
        return

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

    print(f"Loaded {len(chunks)} email chunks")

    # Test thread analyzer
    analyzer = EmailThreadAnalyzer()

    print("\nGrouping emails into threads...")
    threads = analyzer.group_by_thread(chunks)

    print(f"Found {len(threads)} threads")

    # Print statistics
    analyzer.print_thread_stats()

    # Show sample threads
    print("\n" + "="*60)
    print("SAMPLE THREADS")
    print("="*60)

    # Show top 5 longest threads
    sorted_threads = sorted(threads.items(),
                          key=lambda x: len(x[1]),
                          reverse=True)

    for i, (thread_key, thread_emails) in enumerate(sorted_threads[:5], 1):
        print(f"\n{i}. Thread: '{thread_key}' ({len(thread_emails)} emails)")
        print(f"   First email: {thread_emails[0].meta.get('date', 'Unknown date')}")
        print(f"   Last email: {thread_emails[-1].meta.get('date', 'Unknown date')}")
        print(f"   Primary sender: {thread_emails[0].meta.get('sender_name', 'Unknown')}")

    # Test subject normalization
    print("\n" + "="*60)
    print("SUBJECT NORMALIZATION TEST")
    print("="*60)

    test_subjects = [
        "[Primo] Research Assistant Question",
        "Re: [Primo] Research Assistant Question",
        "Re: Re: [Primo] Research Assistant Question",
        "Fwd: [Primo] Research Assistant Question",
        "[ALMA-L] Configuration Issue",
    ]

    for subject in test_subjects:
        normalized = analyzer.normalize_subject(subject)
        print(f"{subject}")
        print(f"  → {normalized}\n")

    print("\n✅ Thread analyzer test complete!")


if __name__ == "__main__":
    main()
