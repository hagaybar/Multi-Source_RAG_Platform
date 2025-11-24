#!/usr/bin/env python3
"""
Email Category Discovery Script

Discovers email categories from existing data using clustering and pattern analysis.

Usage:
    python scripts/categorization/category_discovery.py --project data/projects/Primo_List
"""

import sys
import json
import random
import argparse
import re
import numpy as np
from pathlib import Path
from collections import Counter
from datetime import datetime
from typing import List, Dict, Tuple, Set

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sklearn.feature_extraction.text import TfidfVectorizer

from scripts.core.project_manager import ProjectManager
from scripts.chunking.models import Chunk
from scripts.api_clients.openai.completer import OpenAICompleter

# Comprehensive stopwords for email categorization
EMAIL_STOPWORDS = {
    # Common English stopwords
    'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'your',
    'from', 'that', 'have', 'this', 'will', 'with', 'they', 'what', 'been',
    'were', 'there', 'their', 'would', 'which', 'when', 'where', 'about',
    'than', 'into', 'through', 'some', 'these', 'only', 'other', 'such',
    'them', 'then', 'also', 'does', 'each', 'more', 'most', 'over', 'very',

    # Email-specific words
    'email', 'sent', 'subject', 'date', 'message', 'list', 'thread', 'reply',
    'forward', 'via', 'thanks', 'regards', 'best', 'sincerely',

    # Mailing list generic words
    'primo', 'list', 'mailing', 'listserv', 'group', 'discussion',

    # Time-related (not topics)
    'today', 'yesterday', 'tomorrow', 'week', 'month', 'year', 'days',
    'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
    'january', 'february', 'march', 'april', 'june', 'july', 'august',
    'september', 'october', 'november', 'december',
}

# Patterns to filter out (regex)
FILTER_PATTERNS = [
    r'^\[.*?\]$',  # Mailing list tags: [primo], [alma-l], [igelu]
    r'^\d{4}$',    # Years: 2024, 2025
    r'^\d{1,2}[-/]\d{1,2}$',  # Dates: 05-26, 11/22
    r'^re:?$',     # Reply prefix
    r'^fwd:?$',    # Forward prefix
    r'^http',      # URLs
    r'@',          # Email addresses
]


class CategoryDiscovery:
    """Discovers email categories from existing embeddings using clustering."""

    def __init__(self, project: ProjectManager):
        self.project = project
        self.email_chunks = []
        self.category_mapping = {}  # {cluster_id: category_name}
        self.category_centroids = {}  # {category_name: centroid_embedding}
        self.category_counts = {}  # {category_name: count}
        self.rules = {}  # {category_name: rules_dict}
        self.person_names_blocklist = set()  # Person names to exclude from keywords

    def _extract_person_names(self) -> Set[str]:
        """Extract person names from sender fields to create blocklist.

        This prevents names like 'Stacey', 'Ganor', 'Tamar' from being keywords.
        """
        names = set()

        for chunk in self.email_chunks:
            sender_name = chunk.meta.get("sender_name", "")

            # Common patterns: "FirstName LastName via Primo"
            # Extract individual words from sender names
            if sender_name and sender_name != "Unknown":
                # Remove "via Primo", "via ALMA-L", etc.
                sender_name = re.sub(r'\s+via\s+.*$', '', sender_name, flags=re.IGNORECASE)

                # Split into words and lowercase
                words = sender_name.split()
                for word in words:
                    # Clean and add to blocklist
                    clean_word = word.strip('.,;:()[]{}').lower()
                    if len(clean_word) > 2:  # Avoid initials
                        names.add(clean_word)

        return names

    def _is_valid_keyword(self, word: str) -> bool:
        """Check if a word is a valid keyword (not noise).

        Filters out:
        - Mailing list tags: [primo], [alma-l]
        - Person names from senders
        - Stopwords
        - Years, dates
        - Email artifacts (re:, fwd:, via)
        """
        word = word.lower().strip()

        # Check length
        if len(word) < 4:
            return False

        # Check stopwords
        if word in EMAIL_STOPWORDS:
            return False

        # Check person names blocklist
        if word in self.person_names_blocklist:
            return False

        # Check patterns
        for pattern in FILTER_PATTERNS:
            if re.match(pattern, word, re.IGNORECASE):
                return False

        # Additional checks for common noise
        # - All digits
        if word.isdigit():
            return False

        # - Single characters repeated (e.g., "aaaa")
        if len(set(word)) == 1:
            return False

        return True

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract valid keywords from text.

        Args:
            text: Subject line or body text

        Returns:
            List of valid keywords (filtered)
        """
        # Split into words
        words = text.lower().split()

        # Filter and clean
        keywords = []
        for word in words:
            # Remove punctuation from edges
            clean_word = word.strip('.,;:!?()[]{}"\'`')

            # Check if valid
            if self._is_valid_keyword(clean_word):
                keywords.append(clean_word)

        return keywords

    def _is_system_artifact(self, chunk: Chunk) -> bool:
        """Detect and filter email system artifacts.

        Examples to filter:
        - Outlook reaction notifications
        - Auto-replies
        - Email system messages
        - Very short emails (likely noise)

        Args:
            chunk: Email chunk to check

        Returns:
            True if this is a system artifact, False otherwise
        """
        text = chunk.text.lower()
        subject = chunk.meta.get('subject', '').lower()

        # Reaction notifications
        if 'reacted to your message' in text:
            return True
        if 'outlook-1.cdn.office.net/assets/reaction' in text:
            return True
        if '<https://outlook' in text and 'reaction' in text:
            return True

        # Auto-replies and out-of-office
        if 'out of office' in subject or 'automatic reply' in subject:
            return True
        if 'auto-reply' in subject or 'autoreply' in subject:
            return True

        # Very short emails (likely system messages or fragments)
        if len(chunk.text.strip()) < 50:
            return True

        # Email notification patterns
        if 'has been added to' in text and len(chunk.text) < 150:
            return True

        return False

    def load_email_chunks(self) -> List[Chunk]:
        """Load all email chunks with embeddings from project."""
        print("\n" + "="*60)
        print("Loading Email Chunks")
        print("="*60)

        # Find metadata file
        metadata_path = self.project.get_metadata_path("outlook_eml")

        if not metadata_path.exists():
            print(f"❌ Metadata not found: {metadata_path}")
            return []

        print(f"Loading from: {metadata_path}")

        # Load metadata
        chunks = []
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    meta = json.loads(line)

                    # Create chunk object
                    chunk = Chunk(
                        id=meta.get("id", ""),
                        doc_id=meta.get("doc_id", ""),
                        text=meta.get("text", ""),
                        token_count=meta.get("token_count", 0),
                        meta=meta
                    )
                    chunks.append(chunk)

        print(f"✓ Loaded {len(chunks)} chunks")

        # Load embeddings from vector DB
        print("\nLoading embeddings from vector DB...")
        try:
            from scripts.retrieval.base import FaissRetriever

            # Get paths for FAISS index
            index_path = self.project.root_dir / "output" / "faiss" / "outlook_eml.faiss"
            metadata_path = self.project.get_metadata_path("outlook_eml")

            if not index_path.exists():
                print(f"⚠️ FAISS index not found: {index_path}")
                return chunks

            retriever = FaissRetriever(index_path, metadata_path)

            # Get all embeddings
            for i, chunk in enumerate(chunks):
                if i < len(retriever.metadata):
                    # Match by doc_id or id
                    matching_meta = next(
                        (m for m in retriever.metadata if m.get("id") == chunk.id),
                        None
                    )
                    if matching_meta:
                        # Get embedding from FAISS
                        emb_idx = retriever.metadata.index(matching_meta)
                        embedding = retriever.index.reconstruct(emb_idx)
                        chunk.embedding = embedding

            # Filter chunks without embeddings
            chunks_with_emb = [c for c in chunks if hasattr(c, 'embedding') and c.embedding is not None]

            # Filter system artifacts (Task 1.3 - Phase 1)
            filtered_chunks = [c for c in chunks_with_emb if not self._is_system_artifact(c)]

            removed = len(chunks_with_emb) - len(filtered_chunks)
            if removed > 0:
                print(f"✓ Filtered {removed} system artifacts ({removed/len(chunks_with_emb)*100:.1f}%)")

            print(f"✓ Loaded embeddings for {len(filtered_chunks)} chunks")

            return filtered_chunks

        except Exception as e:
            print(f"⚠️ Could not load embeddings: {e}")
            print("Proceeding with metadata-only analysis...")
            return chunks

    def cluster_embeddings(self, n_categories: int = 7) -> Tuple[Dict, np.ndarray]:
        """Cluster email embeddings to discover natural categories."""
        print("\n" + "="*60)
        print(f"Clustering into {n_categories} Categories")
        print("="*60)

        # Extract embeddings
        embeddings_list = []
        valid_chunks = []

        for chunk in self.email_chunks:
            if hasattr(chunk, 'embedding') and chunk.embedding is not None:
                embeddings_list.append(chunk.embedding)
                valid_chunks.append(chunk)

        if not embeddings_list:
            print("❌ No embeddings available for clustering")
            return {}, np.array([])

        embeddings = np.array(embeddings_list)
        print(f"Clustering {len(embeddings)} embeddings...")

        # Use K-means clustering
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=n_categories, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)

        print(f"✓ Clustering complete")

        # Assign cluster labels to chunks
        for chunk, label in zip(valid_chunks, cluster_labels):
            chunk.meta["cluster_id"] = int(label)

        # Compute cluster centroids
        centroids = {}
        for cluster_id in range(n_categories):
            cluster_embeddings = embeddings[cluster_labels == cluster_id]
            if len(cluster_embeddings) > 0:
                centroid = np.mean(cluster_embeddings, axis=0)
                centroids[cluster_id] = centroid

        return centroids, cluster_labels

    def analyze_cluster(self, cluster_id: int) -> Dict:
        """Analyze a cluster to understand what it represents."""
        cluster_emails = [c for c in self.email_chunks if c.meta.get("cluster_id") == cluster_id]

        if not cluster_emails:
            return None

        print(f"\n{'='*60}")
        print(f"Cluster {cluster_id}: {len(cluster_emails)} emails")
        print(f"{'='*60}")

        # 1. Most common subject keywords (using improved filtering)
        subject_words = []
        for chunk in cluster_emails:
            subject = chunk.meta.get("subject", "")
            keywords = self._extract_keywords(subject)
            subject_words.extend(keywords)

        common_subjects = Counter(subject_words).most_common(15)
        print("\n🔹 Top Subject Keywords (filtered):")
        for word, count in common_subjects[:10]:
            percentage = (count / len(cluster_emails)) * 100
            print(f"   '{word}': {count} times ({percentage:.1f}%)")

        # 2. Sample email subjects
        print("\n🔹 Sample Email Subjects:")
        sample = random.sample(cluster_emails, min(5, len(cluster_emails)))
        for chunk in sample:
            subject = chunk.meta.get("subject", "")[:80]
            date = chunk.meta.get("date", "")[:10]
            print(f"   [{date}] {subject}")

        # 3. Body text patterns (using improved filtering)
        body_words = []
        for chunk in cluster_emails:
            text = chunk.text[:200]  # First 200 chars
            keywords = self._extract_keywords(text)
            body_words.extend(keywords)

        common_body = Counter(body_words).most_common(10)
        print("\n🔹 Common Body Keywords (filtered):")
        for word, count in common_body[:5]:
            print(f"   '{word}': {count} times")

        # 4. Sender patterns (informational only, not used as keywords)
        senders = [c.meta.get("sender_name", "Unknown") for c in cluster_emails]
        sender_counts = Counter(senders).most_common(5)
        print("\n🔹 Top Senders (informational):")
        for sender, count in sender_counts:
            print(f"   {sender}: {count} emails")

        # 5. Temporal pattern
        dates = [c.meta.get("date", "")[:10] for c in cluster_emails if c.meta.get("date")]
        if dates:
            print(f"\n🔹 Date Range: {min(dates)} to {max(dates)}")

        return {
            "cluster_id": cluster_id,
            "size": len(cluster_emails),
            "top_keywords": [w for w, _ in common_subjects[:15]],
            "sample_subjects": [c.meta.get("subject") for c in sample],
            "common_body_words": [w for w, _ in common_body[:10]]
        }

    def _llm_name_cluster(self, cluster_analysis: Dict, used_names: List[str]) -> str:
        """Use LLM to name a cluster based on its analysis.

        Args:
            cluster_analysis: Dict with cluster patterns (from analyze_cluster)
            used_names: List of already-used category names to avoid duplicates

        Returns:
            Category name (1-3 words)
        """
        # Prepare prompt
        prompt = f"""Analyze this email cluster and suggest a SHORT, SPECIFIC category name (1-3 words):

Cluster size: {cluster_analysis['size']} emails
Top subject keywords: {', '.join(cluster_analysis['top_keywords'][:10])}
Sample email subjects:
{chr(10).join(f'  - {subj[:100]}' for subj in cluster_analysis['sample_subjects'][:5])}

Common body keywords: {', '.join(cluster_analysis['common_body_words'][:10])}

Suggest ONE category name that captures what these emails are about.

Examples of good category names:
- Announcements
- Bug Reports
- Feature Requests
- Technical Questions
- Configuration Help
- Performance Issues
- Release Updates

AVOID these already-used names: {', '.join(used_names) if used_names else 'none'}

Return ONLY the category name (1-3 words), nothing else."""

        try:
            completer = OpenAICompleter(model_name="gpt-3.5-turbo")
            response = completer.get_completion(
                prompt=prompt,
                temperature=0.3,
                max_tokens=20
            )

            # Clean response
            category_name = response.strip().strip('"\'').strip()

            # If it's still a duplicate, append cluster ID
            if category_name in used_names:
                category_name = f"{category_name} ({cluster_analysis['cluster_id']})"

            print(f"  [LLM suggested: {category_name}]")
            return category_name

        except Exception as e:
            print(f"  ⚠️ LLM naming failed: {e}")
            # Fallback: use top keyword + cluster ID
            fallback = f"Category_{cluster_analysis['cluster_id']}"
            return fallback

    def name_categories(self, interactive: bool = True, use_llm: bool = True) -> Dict[int, str]:
        """Category naming (interactive or automatic).

        Args:
            interactive: If True, prompt for category names. If False, use suggestions.
            use_llm: If True and non-interactive, use LLM to name clusters.
        """
        print("\n" + "="*60)
        print("CATEGORY NAMING")
        print("="*60)
        if interactive:
            print("\nBased on the cluster analysis, please name each category.")
            print("Suggested names based on patterns will be shown.\n")
        else:
            if use_llm:
                print("\nUsing LLM to intelligently name categories...\n")
            else:
                print("\nAuto-naming categories based on discovered patterns...\n")

        # Get unique cluster IDs
        cluster_ids = sorted(set(c.meta.get("cluster_id") for c in self.email_chunks if "cluster_id" in c.meta))

        category_mapping = {}
        used_names = []  # Track used names to avoid duplicates

        for cluster_id in cluster_ids:
            # Suggest name based on keywords (using improved filtering)
            cluster_emails = [c for c in self.email_chunks if c.meta.get("cluster_id") == cluster_id]
            subject_words = []
            for chunk in cluster_emails:
                subject = chunk.meta.get("subject", "")
                keywords = self._extract_keywords(subject)
                subject_words.extend(keywords)

            top_words = [w for w, _ in Counter(subject_words).most_common(3)]

            # Suggest category name
            suggestions = []
            if any(w in top_words for w in ['bug', 'error', 'issue', 'problem']):
                suggestions.append("Bug Reports")
            if any(w in top_words for w in ['feature', 'request', 'enhancement']):
                suggestions.append("Feature Requests")
            if any(w in top_words for w in ['question', 'help', 'how']):
                suggestions.append("Questions")
            if any(w in top_words for w in ['release', 'announce', 'update']):
                suggestions.append("Announcements")
            if any(w in top_words for w in ['config', 'setup', 'install']):
                suggestions.append("Configuration")

            suggestion = suggestions[0] if suggestions else "Discussion"

            print(f"\nCluster {cluster_id} ({len(cluster_emails)} emails)")
            print(f"  Top keywords: {', '.join(top_words[:5])}")

            if interactive:
                # Interactive mode: show suggestion and prompt
                print(f"  Suggested: {suggestion}")
                name = input(f"  Enter category name [default: {suggestion}]: ").strip()
                if not name:
                    name = suggestion
            elif use_llm and not interactive:
                # Auto + LLM mode: use LLM to name cluster
                # Create cluster analysis for LLM
                cluster_analysis = {
                    "cluster_id": cluster_id,
                    "size": len(cluster_emails),
                    "top_keywords": top_words[:15],
                    "sample_subjects": [c.meta.get("subject", "")[:100] for c in cluster_emails[:5]],
                    "common_body_words": [w for w, _ in Counter(
                        [w for c in cluster_emails for w in c.text.lower().split() if len(w) > 4]
                    ).most_common(10)]
                }
                name = self._llm_name_cluster(cluster_analysis, used_names)
            else:
                # Auto without LLM: use simple suggestion
                print(f"  Suggested: {suggestion}")
                name = suggestion
                # Check for duplicates
                if name in used_names:
                    name = f"{name}_{cluster_id}"
                    print(f"  [Duplicate detected, renamed to: {name}]")

            category_mapping[cluster_id] = name
            used_names.append(name)
            print(f"  ✓ Named as: '{name}'")

        return category_mapping

    def extract_rules(self) -> Dict[str, Dict]:
        """Extract categorization rules using TF-IDF for distinctive keywords."""
        print("\n" + "="*60)
        print("Extracting Categorization Rules (TF-IDF)")
        print("="*60)

        rules = {}

        for cluster_id, category_name in self.category_mapping.items():
            cluster_emails = [c for c in self.email_chunks
                             if c.meta.get("cluster_id") == cluster_id]

            if not cluster_emails:
                continue

            # Prepare texts for TF-IDF
            subject_texts = [c.meta.get("subject", "") for c in cluster_emails]
            body_texts = [c.text for c in cluster_emails]

            # TF-IDF for subject keywords (Task 1.2 - Phase 1)
            subject_keywords = []
            if subject_texts and len([s for s in subject_texts if s.strip()]) > 1:
                try:
                    vectorizer_subj = TfidfVectorizer(
                        max_features=10,
                        ngram_range=(1, 2),  # Unigrams + bigrams (e.g., "research assistant")
                        stop_words=list(EMAIL_STOPWORDS),
                        min_df=2  # Must appear in at least 2 emails
                    )

                    tfidf_matrix = vectorizer_subj.fit_transform(subject_texts)
                    feature_names = vectorizer_subj.get_feature_names_out()

                    # Get average TF-IDF score per feature
                    scores = tfidf_matrix.mean(axis=0).A1
                    top_indices = scores.argsort()[-10:][::-1]
                    subject_keywords = [feature_names[i] for i in top_indices]
                except Exception as e:
                    # Fallback to simple filtering if TF-IDF fails
                    subject_keywords = []

            # TF-IDF for body keywords
            body_keywords = []
            if body_texts and len([b for b in body_texts if b.strip()]) > 1:
                try:
                    vectorizer_body = TfidfVectorizer(
                        max_features=10,
                        ngram_range=(1, 2),
                        stop_words=list(EMAIL_STOPWORDS),
                        min_df=2
                    )

                    tfidf_matrix = vectorizer_body.fit_transform(body_texts)
                    feature_names = vectorizer_body.get_feature_names_out()

                    scores = tfidf_matrix.mean(axis=0).A1
                    top_indices = scores.argsort()[-10:][::-1]
                    body_keywords = [feature_names[i] for i in top_indices]
                except Exception as e:
                    # Fallback to simple filtering if TF-IDF fails
                    body_keywords = []

            rules[category_name] = {
                "cluster_id": cluster_id,
                "subject_keywords": subject_keywords,
                "body_keywords": body_keywords,
                "confidence": 0.65,
                "sample_size": len(cluster_emails),
                "extraction_method": "tfidf"  # Track which method was used
            }

            print(f"✓ {category_name}: {len(subject_keywords)} subject, "
                  f"{len(body_keywords)} body keywords (TF-IDF)")

        return rules

    def compute_centroids(self) -> Tuple[Dict, Dict]:
        """Compute category centroids from clustered emails."""
        print("\n" + "="*60)
        print("Computing Category Centroids")
        print("="*60)

        centroids = {}
        counts = {}

        for cluster_id, category_name in self.category_mapping.items():
            cluster_emails = [c for c in self.email_chunks if c.meta.get("cluster_id") == cluster_id]

            # Get embeddings
            embeddings = []
            for chunk in cluster_emails:
                if hasattr(chunk, 'embedding') and chunk.embedding is not None:
                    embeddings.append(chunk.embedding)

            if embeddings:
                centroid = np.mean(embeddings, axis=0)
                centroids[category_name] = centroid
                counts[category_name] = len(embeddings)

                print(f"✓ {category_name}: {len(embeddings)} emails")

        return centroids, counts

    def save_results(self, output_path: Path):
        """Save discovered categories, rules, and centroids."""
        print("\n" + "="*60)
        print("Saving Results")
        print("="*60)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare data
        data = {
            "discovery_date": datetime.now().isoformat(),
            "project": str(self.project.root_dir),
            "total_emails": len(self.email_chunks),
            "category_mapping": self.category_mapping,
            "rules": self.rules,
            "centroids": {k: v.tolist() for k, v in self.category_centroids.items()},
            "counts": self.category_counts
        }

        # Save JSON
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"✓ Saved to: {output_path}")

        # Print summary
        print("\n" + "="*60)
        print("DISCOVERY SUMMARY")
        print("="*60)
        print(f"\nTotal emails analyzed: {len(self.email_chunks)}")
        print(f"Categories discovered: {len(self.category_mapping)}")
        print("\nCategories:")

        for category, count in sorted(self.category_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(self.email_chunks)) * 100
            keywords = self.rules[category]["subject_keywords"][:5]
            print(f"  • {category}: {count} emails ({percentage:.1f}%)")
            print(f"    Keywords: {', '.join(keywords)}")

    def run(self, n_categories: int = 7, output_path: Path = None, interactive: bool = True):
        """Run complete discovery process.

        Args:
            n_categories: Number of categories to discover
            output_path: Where to save results
            interactive: If True, prompt user for input. If False, auto-name categories.
        """
        # Step 1: Load data
        self.email_chunks = self.load_email_chunks()

        if not self.email_chunks:
            print("❌ No emails found")
            return

        # Step 1.5: Extract person names for filtering
        print("\n" + "="*60)
        print("Building Keyword Filters")
        print("="*60)
        self.person_names_blocklist = self._extract_person_names()
        print(f"✓ Extracted {len(self.person_names_blocklist)} person names to exclude from keywords")
        print(f"  Sample names: {', '.join(list(self.person_names_blocklist)[:10])}")

        # Step 2: Cluster embeddings
        if any(hasattr(c, 'embedding') for c in self.email_chunks):
            cluster_centroids, cluster_labels = self.cluster_embeddings(n_categories)
        else:
            print("⚠️ No embeddings available, skipping clustering")
            return

        # Step 3: Analyze clusters
        print("\n" + "="*60)
        print("CLUSTER ANALYSIS")
        print("="*60)
        print("Analyzing each cluster to understand patterns...\n")

        cluster_ids = sorted(set(c.meta.get("cluster_id") for c in self.email_chunks if "cluster_id" in c.meta))

        for cluster_id in cluster_ids:
            self.analyze_cluster(cluster_id)
            if interactive:
                input("\n[Press Enter to continue...]")

        # Step 4: Name categories
        self.category_mapping = self.name_categories(interactive=interactive)

        # Step 5: Extract rules
        self.rules = self.extract_rules()

        # Step 6: Compute centroids
        self.category_centroids, self.category_counts = self.compute_centroids()

        # Step 7: Save results
        if output_path is None:
            output_path = Path("data/categories/discovered_categories.json")

        self.save_results(output_path)

        print("\n✅ Category discovery complete!")
        print(f"\nNext steps:")
        print("1. Review the discovered categories in: {output_path}")
        print("2. Implement EmailCategorizer to use these categories")
        print("3. Integrate into ingestion pipeline")
        print("4. Test with new emails")


def main():
    parser = argparse.ArgumentParser(description="Discover email categories from existing data")
    parser.add_argument("--project", type=str, default="data/projects/Primo_List",
                       help="Path to project directory")
    parser.add_argument("--n-categories", type=int, default=7,
                       help="Number of categories to discover")
    parser.add_argument("--output", type=str, default="data/categories/discovered_categories.json",
                       help="Output path for results")
    parser.add_argument("--auto", action="store_true",
                       help="Run in non-interactive mode (auto-name categories)")

    args = parser.parse_args()

    print("="*60)
    print("  EMAIL CATEGORY DISCOVERY")
    print("="*60)
    print(f"\nProject: {args.project}")
    print(f"Target categories: {args.n_categories}")
    print(f"Output: {args.output}")

    # Load project
    project_path = Path(args.project)
    if not project_path.exists():
        print(f"❌ Project not found: {project_path}")
        return 1

    project = ProjectManager(project_path)

    # Run discovery
    discovery = CategoryDiscovery(project)
    discovery.run(n_categories=args.n_categories, output_path=Path(args.output), interactive=not args.auto)

    return 0


if __name__ == "__main__":
    sys.exit(main())
