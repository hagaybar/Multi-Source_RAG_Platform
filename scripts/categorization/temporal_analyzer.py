#!/usr/bin/env python3
"""
Temporal Topic Analyzer - Detect time-bound topics in email collections

Analyzes how topics evolve over time and detects temporal patterns like:
- Time-bound topics (e.g., "June 2025 Release Issues")
- Topic emergence and decline
- Seasonal patterns
- Event-driven topics
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from collections import Counter, defaultdict

from scripts.chunking.models import Chunk


class TemporalTopicAnalyzer:
    """Detects temporal patterns in email topics.

    Features:
    - Analyze topic evolution over time
    - Detect time-bound topics
    - Identify trending topics
    - Monthly keyword extraction
    - Temporal topic shifts
    """

    def __init__(self, time_unit: str = 'month'):
        """Initialize temporal analyzer.

        Args:
            time_unit: Time unit for grouping ('day', 'week', 'month', 'quarter')
        """
        self.time_unit = time_unit
        self.temporal_data = None
        self.temporal_topics = {}

    def analyze_temporal_evolution(self, emails: List[Chunk],
                                   categories: Optional[Dict] = None) -> Dict:
        """Analyze how topics evolve over time.

        Args:
            emails: List of email chunks
            categories: Optional dict of category mappings

        Returns:
            Dict with temporal analysis results
        """
        # Convert to DataFrame for easier analysis
        df = self._create_temporal_dataframe(emails)

        if df.empty:
            print("Warning: No valid email data for temporal analysis")
            return {}

        # Group by time period and category
        df['period'] = self._group_by_period(df['date'])

        # Analyze each category
        temporal_topics = {}

        for category in df['category'].unique():
            cat_df = df[df['category'] == category]

            # Monthly keyword trends
            monthly_keywords = self._extract_monthly_keywords(cat_df)

            # Detect temporal shifts
            has_shift = self._has_temporal_shift(monthly_keywords)

            # Calculate topic velocity (growth/decline)
            velocity = self._calculate_topic_velocity(cat_df)

            # Detect time-bound periods
            time_bound_periods = self._detect_time_bound_periods(cat_df)

            temporal_topics[category] = {
                'monthly_keywords': monthly_keywords,
                'has_temporal_shift': has_shift,
                'velocity': velocity,
                'time_bound_periods': time_bound_periods,
                'peak_period': self._find_peak_period(cat_df),
                'total_emails': len(cat_df),
                'active_periods': cat_df['period'].nunique(),
            }

        self.temporal_topics = temporal_topics
        self.temporal_data = df

        return temporal_topics

    def _create_temporal_dataframe(self, emails: List[Chunk]) -> pd.DataFrame:
        """Create DataFrame with temporal information.

        Args:
            emails: List of email chunks

        Returns:
            DataFrame with date, category, text columns
        """
        data = []

        for email in emails:
            date_str = email.meta.get('date', '')
            category = email.meta.get('category', 'Unknown')
            text = email.text[:200]  # Sample text

            # Parse date
            try:
                if date_str:
                    date = pd.to_datetime(date_str)
                else:
                    continue  # Skip emails without dates
            except:
                continue

            data.append({
                'date': date,
                'category': category,
                'text': text,
                'subject': email.meta.get('subject', ''),
                'sender': email.meta.get('sender_name', ''),
            })

        return pd.DataFrame(data)

    def _group_by_period(self, dates: pd.Series) -> pd.Series:
        """Group dates by time period.

        Args:
            dates: Series of datetime objects

        Returns:
            Series of period strings
        """
        if self.time_unit == 'day':
            return dates.dt.to_period('D').astype(str)
        elif self.time_unit == 'week':
            return dates.dt.to_period('W').astype(str)
        elif self.time_unit == 'month':
            return dates.dt.to_period('M').astype(str)
        elif self.time_unit == 'quarter':
            return dates.dt.to_period('Q').astype(str)
        else:
            return dates.dt.to_period('M').astype(str)

    def _extract_monthly_keywords(self, cat_df: pd.DataFrame) -> Dict[str, List[str]]:
        """Extract top keywords for each month.

        Args:
            cat_df: DataFrame for a specific category

        Returns:
            Dict mapping period → list of top keywords
        """
        monthly_keywords = {}

        for period in cat_df['period'].unique():
            period_df = cat_df[cat_df['period'] == period]
            texts = period_df['text'].tolist()

            keywords = self._extract_keywords_from_texts(texts)
            monthly_keywords[period] = keywords

        return monthly_keywords

    def _extract_keywords_from_texts(self, texts: List[str],
                                    top_n: int = 5) -> List[str]:
        """Extract top keywords from text list using TF-IDF.

        Args:
            texts: List of text strings
            top_n: Number of top keywords to return

        Returns:
            List of top keywords
        """
        if not texts:
            return []

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            vectorizer = TfidfVectorizer(
                max_features=top_n,
                ngram_range=(1, 2),
                stop_words='english',
                min_df=1
            )

            tfidf_matrix = vectorizer.fit_transform(texts)
            keywords = vectorizer.get_feature_names_out().tolist()

            return keywords
        except Exception as e:
            # Fallback: simple word frequency
            words = []
            for text in texts:
                words.extend(text.lower().split())

            # Count and return top N
            word_counts = Counter(words)
            return [word for word, count in word_counts.most_common(top_n)]

    def _has_temporal_shift(self, monthly_keywords: Dict[str, List[str]]) -> bool:
        """Detect if topic keywords change significantly over time.

        Args:
            monthly_keywords: Dict mapping period → keywords

        Returns:
            True if significant temporal shift detected
        """
        if len(monthly_keywords) < 2:
            return False

        # Compare keyword overlap between consecutive periods
        periods = sorted(monthly_keywords.keys())
        overlaps = []

        for i in range(len(periods) - 1):
            keywords1 = set(monthly_keywords[periods[i]])
            keywords2 = set(monthly_keywords[periods[i+1]])

            if keywords1 and keywords2:
                # Jaccard similarity
                overlap = len(keywords1 & keywords2) / len(keywords1 | keywords2)
                overlaps.append(overlap)

        # If average overlap < 0.5, topic is shifting
        return np.mean(overlaps) < 0.5 if overlaps else False

    def _calculate_topic_velocity(self, cat_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate topic growth/decline velocity.

        Args:
            cat_df: DataFrame for a specific category

        Returns:
            Dict with velocity metrics
        """
        # Group by period and count emails
        period_counts = cat_df.groupby('period').size()

        if len(period_counts) < 2:
            return {'trend': 'stable', 'velocity': 0.0}

        # Calculate trend
        periods = list(range(len(period_counts)))
        counts = period_counts.values

        # Linear regression to find trend
        slope = np.polyfit(periods, counts, 1)[0]

        # Classify trend
        if slope > 1:
            trend = 'growing'
        elif slope < -1:
            trend = 'declining'
        else:
            trend = 'stable'

        return {
            'trend': trend,
            'velocity': float(slope),
            'avg_emails_per_period': float(np.mean(counts)),
            'peak_count': int(np.max(counts)),
            'min_count': int(np.min(counts)),
        }

    def _detect_time_bound_periods(self, cat_df: pd.DataFrame,
                                   threshold: float = 2.0) -> List[str]:
        """Detect periods with unusually high activity (time-bound topics).

        Args:
            cat_df: DataFrame for a specific category
            threshold: Standard deviations above mean to consider time-bound

        Returns:
            List of period strings with high activity
        """
        # Count emails per period
        period_counts = cat_df.groupby('period').size()

        if len(period_counts) < 3:
            return []

        # Calculate mean and std
        mean_count = period_counts.mean()
        std_count = period_counts.std()

        # Find periods above threshold
        time_bound = []
        for period, count in period_counts.items():
            if count > mean_count + (threshold * std_count):
                time_bound.append(period)

        return time_bound

    def _find_peak_period(self, cat_df: pd.DataFrame) -> Tuple[str, int]:
        """Find period with highest activity.

        Args:
            cat_df: DataFrame for a specific category

        Returns:
            Tuple of (period, email_count)
        """
        period_counts = cat_df.groupby('period').size()

        if period_counts.empty:
            return ("Unknown", 0)

        peak_period = period_counts.idxmax()
        peak_count = period_counts.max()

        return (str(peak_period), int(peak_count))

    def get_trending_topics(self, min_velocity: float = 1.0) -> List[Tuple[str, float]]:
        """Get topics that are trending (growing).

        Args:
            min_velocity: Minimum velocity to consider trending

        Returns:
            List of (category, velocity) tuples sorted by velocity
        """
        if not self.temporal_topics:
            return []

        trending = [
            (category, data['velocity']['velocity'])
            for category, data in self.temporal_topics.items()
            if data['velocity']['trend'] == 'growing'
            and data['velocity']['velocity'] >= min_velocity
        ]

        return sorted(trending, key=lambda x: x[1], reverse=True)

    def get_time_bound_topics(self) -> Dict[str, List[str]]:
        """Get topics that are time-bound (spike in specific periods).

        Returns:
            Dict mapping category → list of time-bound periods
        """
        if not self.temporal_topics:
            return {}

        time_bound = {
            category: data['time_bound_periods']
            for category, data in self.temporal_topics.items()
            if data['time_bound_periods']
        }

        return time_bound

    def print_temporal_analysis(self):
        """Print temporal analysis results."""
        if not self.temporal_topics:
            print("No temporal analysis available. Run analyze_temporal_evolution() first.")
            return

        print("\n" + "="*60)
        print("TEMPORAL TOPIC ANALYSIS")
        print("="*60)

        for category, data in sorted(self.temporal_topics.items(),
                                    key=lambda x: x[1]['total_emails'],
                                    reverse=True):
            print(f"\n{category}:")
            print(f"  Total emails: {data['total_emails']}")
            print(f"  Active periods: {data['active_periods']}")
            print(f"  Peak period: {data['peak_period'][0]} ({data['peak_period'][1]} emails)")

            velocity = data['velocity']
            print(f"  Trend: {velocity['trend']} (velocity: {velocity['velocity']:.2f})")

            if data['time_bound_periods']:
                print(f"  Time-bound periods: {', '.join(data['time_bound_periods'])}")

            if data['has_temporal_shift']:
                print(f"  ⚠️  Topic shifted significantly over time")

        # Print trending topics
        trending = self.get_trending_topics()
        if trending:
            print(f"\n{'='*60}")
            print("TRENDING TOPICS (Growing):")
            for category, velocity in trending:
                print(f"  • {category} (velocity: {velocity:.2f})")

        # Print time-bound topics
        time_bound = self.get_time_bound_topics()
        if time_bound:
            print(f"\n{'='*60}")
            print("TIME-BOUND TOPICS (Activity spikes):")
            for category, periods in time_bound.items():
                print(f"  • {category}: {', '.join(periods)}")

        print("="*60)

    def create_temporal_timeline(self) -> pd.DataFrame:
        """Create timeline showing email counts per category per period.

        Returns:
            DataFrame with periods as rows, categories as columns
        """
        if self.temporal_data is None:
            return pd.DataFrame()

        # Pivot table: periods × categories
        timeline = self.temporal_data.pivot_table(
            values='text',
            index='period',
            columns='category',
            aggfunc='count',
            fill_value=0
        )

        return timeline

    def print_temporal_timeline(self):
        """Print temporal timeline."""
        timeline = self.create_temporal_timeline()

        if timeline.empty:
            print("No timeline data available.")
            return

        print("\n" + "="*60)
        print("TEMPORAL TIMELINE (Email Counts)")
        print("="*60)
        print(timeline.to_string())
        print("="*60)


def main():
    """Test temporal analyzer with sample data."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from scripts.core.project_manager import ProjectManager

    # Test on Primo_List_2
    project = ProjectManager(Path("data/projects/Primo_List_2"))

    print("="*60)
    print("  TEMPORAL TOPIC ANALYZER TEST")
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

    # Add mock categories for testing (in real use, these come from categorization)
    # Distribute randomly for demonstration
    categories = ['Research Support', 'UI Customization', 'Technical Issues', 'Product Updates']
    import random
    for chunk in chunks:
        chunk.meta['category'] = random.choice(categories)

    # Test temporal analyzer
    analyzer = TemporalTopicAnalyzer(time_unit='month')

    print("\nAnalyzing temporal evolution...")
    temporal_topics = analyzer.analyze_temporal_evolution(chunks)

    print(f"Analyzed {len(temporal_topics)} categories")

    # Print analysis
    analyzer.print_temporal_analysis()

    # Print timeline
    analyzer.print_temporal_timeline()

    print("\n✅ Temporal analyzer test complete!")


if __name__ == "__main__":
    main()
