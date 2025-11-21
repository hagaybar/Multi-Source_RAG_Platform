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
import numpy as np
from pathlib import Path
from collections import Counter
from datetime import datetime
from typing import List, Dict, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.core.project_manager import ProjectManager
from scripts.chunking.models import Chunk


class CategoryDiscovery:
    """Discovers email categories from existing embeddings using clustering."""

    def __init__(self, project: ProjectManager):
        self.project = project
        self.email_chunks = []
        self.category_mapping = {}  # {cluster_id: category_name}
        self.category_centroids = {}  # {category_name: centroid_embedding}
        self.category_counts = {}  # {category_name: count}
        self.rules = {}  # {category_name: rules_dict}

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
            index_path = self.project.project_dir / "output" / "embeddings" / "outlook_eml.faiss"
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
            print(f"✓ Loaded embeddings for {len(chunks_with_emb)} chunks")

            return chunks_with_emb

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

        # 1. Most common subject keywords
        subject_words = []
        for chunk in cluster_emails:
            subject = chunk.meta.get("subject", "").lower()
            # Remove common words
            words = [w for w in subject.split() if len(w) > 3 and w not in ['from', 'sent', 'subject', 'date', 'primo', 'list']]
            subject_words.extend(words)

        common_subjects = Counter(subject_words).most_common(15)
        print("\n🔹 Top Subject Keywords:")
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

        # 3. Body text patterns
        body_words = []
        for chunk in cluster_emails:
            text = chunk.text.lower()[:200]  # First 200 chars
            words = [w for w in text.split() if len(w) > 4]
            body_words.extend(words)

        common_body = Counter(body_words).most_common(10)
        print("\n🔹 Common Body Keywords:")
        for word, count in common_body[:5]:
            print(f"   '{word}': {count} times")

        # 4. Sender patterns
        senders = [c.meta.get("sender_name", "Unknown") for c in cluster_emails]
        sender_counts = Counter(senders).most_common(5)
        print("\n🔹 Top Senders:")
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

    def name_categories(self) -> Dict[int, str]:
        """Interactive category naming."""
        print("\n" + "="*60)
        print("CATEGORY NAMING")
        print("="*60)
        print("\nBased on the cluster analysis, please name each category.")
        print("Suggested names based on patterns will be shown.\n")

        # Get unique cluster IDs
        cluster_ids = sorted(set(c.meta.get("cluster_id") for c in self.email_chunks if "cluster_id" in c.meta))

        category_mapping = {}

        for cluster_id in cluster_ids:
            # Suggest name based on keywords
            cluster_emails = [c for c in self.email_chunks if c.meta.get("cluster_id") == cluster_id]
            subject_words = []
            for chunk in cluster_emails:
                subject = chunk.meta.get("subject", "").lower()
                words = [w for w in subject.split() if len(w) > 3]
                subject_words.extend(words)

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
            print(f"  Suggested: {suggestion}")

            name = input(f"  Enter category name [default: {suggestion}]: ").strip()
            if not name:
                name = suggestion

            category_mapping[cluster_id] = name
            print(f"  ✓ Named as: '{name}'")

        return category_mapping

    def extract_rules(self) -> Dict[str, Dict]:
        """Extract categorization rules from discovered patterns."""
        print("\n" + "="*60)
        print("Extracting Categorization Rules")
        print("="*60)

        rules = {}

        for cluster_id, category_name in self.category_mapping.items():
            cluster_emails = [c for c in self.email_chunks if c.meta.get("cluster_id") == cluster_id]

            if not cluster_emails:
                continue

            # Extract subject keywords
            subject_words = []
            body_words = []

            for chunk in cluster_emails:
                subject = chunk.meta.get("subject", "").lower()
                body = chunk.text.lower()

                subject_words.extend([w for w in subject.split() if len(w) > 3])
                body_words.extend([w for w in body.split() if len(w) > 4])

            # Find keywords that appear in >30% of emails
            threshold = len(cluster_emails) * 0.3

            common_subject = [w for w, count in Counter(subject_words).items() if count > threshold]
            common_body = [w for w, count in Counter(body_words).items() if count > threshold]

            rules[category_name] = {
                "cluster_id": cluster_id,
                "subject_keywords": common_subject[:10],
                "body_keywords": common_body[:10],
                "confidence": 0.65,
                "sample_size": len(cluster_emails)
            }

            print(f"✓ {category_name}: {len(common_subject)} subject keywords, {len(common_body)} body keywords")

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
            "project": str(self.project.project_dir),
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

    def run(self, n_categories: int = 7, output_path: Path = None):
        """Run complete discovery process."""
        # Step 1: Load data
        self.email_chunks = self.load_email_chunks()

        if not self.email_chunks:
            print("❌ No emails found")
            return

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
            input("\n[Press Enter to continue...]")

        # Step 4: Name categories
        self.category_mapping = self.name_categories()

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
    discovery.run(n_categories=args.n_categories, output_path=Path(args.output))

    return 0


if __name__ == "__main__":
    sys.exit(main())
