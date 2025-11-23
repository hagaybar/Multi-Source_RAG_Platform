#!/usr/bin/env python3
"""
BERTopic-based Email Category Discovery

Uses state-of-the-art topic modeling instead of K-means.
Automatically determines optimal number of topics.

Usage:
    python scripts/categorization/bertopic_discovery.py --project data/projects/Primo_List
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from umap import UMAP
from hdbscan import HDBSCAN

from scripts.core.project_manager import ProjectManager
from scripts.chunking.models import Chunk
from scripts.api_clients.openai.completer import OpenAICompleter
from scripts.categorization.category_discovery import (
    CategoryDiscovery,
    EMAIL_STOPWORDS
)


class BERTopicCategoryDiscovery(CategoryDiscovery):
    """BERTopic-based category discovery (extends K-means version)."""

    def __init__(self, project: ProjectManager):
        super().__init__(project)
        self.topic_model = None

    def cluster_embeddings_bertopic(self, min_cluster_size: int = 30, use_kmeans: bool = False, n_clusters: int = 5) -> Tuple[Dict, List[int]]:
        """Cluster emails using BERTopic instead of K-means.

        Args:
            min_cluster_size: Minimum emails per topic (default: 30)

        Returns:
            cluster_centroids: Dict mapping cluster_id to centroid embedding
            topics: List of topic assignments per email
        """
        print("\n" + "="*60)
        if use_kmeans:
            print(f"BERTopic Clustering (K-means with {n_clusters} clusters)")
        else:
            print(f"BERTopic Clustering (HDBSCAN auto-detect optimal topics)")
        print("="*60)

        # Extract embeddings and documents
        embeddings_list = []
        documents = []
        valid_chunks = []

        for chunk in self.email_chunks:
            if hasattr(chunk, 'embedding') and chunk.embedding is not None:
                embeddings_list.append(chunk.embedding)
                # Document = subject + preview
                doc = chunk.meta.get('subject', '') + " " + chunk.text[:200]
                documents.append(doc)
                valid_chunks.append(chunk)

        if not embeddings_list:
            print("❌ No embeddings available")
            return {}, []

        embeddings = np.array(embeddings_list)
        print(f"Clustering {len(embeddings)} emails...")

        # Configure BERTopic components
        umap_model = UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric='cosine',
            random_state=42
        )

        # Choose clustering algorithm
        if use_kmeans:
            cluster_model = KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=10
            )
        else:
            cluster_model = HDBSCAN(
                min_cluster_size=min_cluster_size,
                metric='euclidean',
                cluster_selection_method='eom',
                prediction_data=True
            )

        # Adaptive min_df based on dataset size and number of clusters
        # With K-means, BERTopic creates one document per cluster for vectorization
        # So min_df must be < n_clusters to avoid errors
        # Small datasets: min_df=1, Large datasets: min_df=max(1, n_clusters//2)
        if use_kmeans:
            adaptive_min_df = max(1, min(2, n_clusters // 2))
        else:
            adaptive_min_df = max(1, min(5, len(embeddings) // 200))

        vectorizer_model = CountVectorizer(
            stop_words=list(EMAIL_STOPWORDS),
            ngram_range=(1, 3),  # Unigrams, bigrams, trigrams
            min_df=adaptive_min_df,  # Adaptive based on dataset size
            max_df=0.95  # Ignore terms in >95% of documents
        )

        # Create BERTopic model
        self.topic_model = BERTopic(
            embedding_model=None,  # Use pre-computed embeddings!
            umap_model=umap_model,
            hdbscan_model=cluster_model,  # K-means or HDBSCAN
            vectorizer_model=vectorizer_model,
            calculate_probabilities=True if not use_kmeans else False,  # K-means doesn't support probabilities
            verbose=True
        )

        # Fit model
        topics, probs = self.topic_model.fit_transform(documents, embeddings)

        # Get topic info
        topic_info = self.topic_model.get_topic_info()

        num_outliers = sum(1 for t in topics if t == -1)
        num_topics = len(set(topics)) - (1 if num_outliers > 0 else 0)

        print(f"✓ Discovered {num_topics} topics")
        if num_outliers > 0:
            print(f"  Outliers: {num_outliers} emails ({num_outliers/len(topics)*100:.1f}%)")
        else:
            print(f"  Outliers: 0 emails (K-means assigns all documents)")

        # Assign cluster labels to chunks
        for chunk, topic in zip(valid_chunks, topics):
            chunk.meta["cluster_id"] = int(topic)

        # Compute centroids for non-outlier topics
        centroids = {}
        for topic_id in set(topics):
            if topic_id == -1:  # Skip outliers
                continue
            topic_embeddings = embeddings[np.array(topics) == topic_id]
            if len(topic_embeddings) > 0:
                centroid = np.mean(topic_embeddings, axis=0)
                centroids[topic_id] = centroid

        return centroids, topics

    def get_bertopic_keywords(self, topic_id: int, top_n: int = 10) -> List[str]:
        """Get top keywords for a topic from BERTopic."""
        if self.topic_model is None:
            return []

        if topic_id == -1:  # Outliers have no keywords
            return []

        topic_words = self.topic_model.get_topic(topic_id)
        if topic_words:
            return [word for word, _ in topic_words[:top_n]]
        return []

    def run_bertopic(self, min_cluster_size: int = 30,
                    output_path: Path = None, interactive: bool = False,
                    use_kmeans: bool = False, n_clusters: int = 5):
        """Run BERTopic-based discovery process."""

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
        print(f"✓ Extracted {len(self.person_names_blocklist)} person names")

        # Step 2: BERTopic clustering
        cluster_centroids, cluster_labels = self.cluster_embeddings_bertopic(
            min_cluster_size=min_cluster_size,
            use_kmeans=use_kmeans,
            n_clusters=n_clusters
        )

        # Step 3: Analyze clusters (reuse parent method)
        print("\n" + "="*60)
        print("CLUSTER ANALYSIS")
        print("="*60)

        cluster_ids = sorted(set(c.meta.get("cluster_id")
                                for c in self.email_chunks
                                if "cluster_id" in c.meta))

        for cluster_id in cluster_ids:
            if cluster_id == -1:  # Skip outliers in analysis
                continue
            self.analyze_cluster(cluster_id)
            if interactive:
                input("\n[Press Enter to continue...]")

        # Step 4: Name categories (using LLM + BERTopic keywords)
        print("\n" + "="*60)
        print("CATEGORY NAMING (BERTopic + LLM)")
        print("="*60)

        category_mapping = {}
        used_names = []

        for cluster_id in cluster_ids:
            if cluster_id == -1:
                category_mapping[cluster_id] = "Outliers"
                continue

            # Get BERTopic keywords
            bertopic_keywords = self.get_bertopic_keywords(cluster_id, top_n=10)

            # Use LLM naming with BERTopic keywords
            cluster_analysis = {
                "cluster_id": cluster_id,
                "size": len([c for c in self.email_chunks
                           if c.meta.get("cluster_id") == cluster_id]),
                "top_keywords": bertopic_keywords,
                "sample_subjects": [c.meta.get("subject", "")[:100]
                                   for c in self.email_chunks
                                   if c.meta.get("cluster_id") == cluster_id][:5],
                "common_body_words": bertopic_keywords[:5]
            }

            name = self._llm_name_cluster(cluster_analysis, used_names)
            category_mapping[cluster_id] = name
            used_names.append(name)
            print(f"  ✓ Named as: '{name}'")

        self.category_mapping = category_mapping

        # Step 5: Extract rules (using BERTopic keywords + TF-IDF)
        self.rules = self.extract_rules()

        # Step 6: Compute centroids
        self.category_centroids, self.category_counts = self.compute_centroids()

        # Step 7: Save results
        if output_path is None:
            output_path = Path("data/categories/discovered_categories_bertopic.json")

        self.save_results(output_path)

        print("\n✅ BERTopic discovery complete!")

    def create_hierarchical_categories(self, max_fine_topics: int = 15,
                                       n_broad_categories: int = 5,
                                       output_path: Path = None) -> Dict:
        """Create 2-level category hierarchy (broad → specific).

        Args:
            max_fine_topics: Target number of fine-grained topics (default: 15)
            n_broad_categories: Number of broad categories (default: 5)
            output_path: Path to save hierarchy results

        Returns:
            Dict with hierarchy structure
        """
        print("\n" + "="*60)
        print("HIERARCHICAL CATEGORY DISCOVERY")
        print("="*60)
        print(f"Goal: {n_broad_categories} broad categories, ~{max_fine_topics} specific topics")

        # Step 1: Load data
        self.email_chunks = self.load_email_chunks()

        if not self.email_chunks:
            print("❌ No emails found")
            return {}

        # Extract person names for filtering
        print("\n" + "="*60)
        print("Building Keyword Filters")
        print("="*60)
        self.person_names_blocklist = self._extract_person_names()
        print(f"✓ Extracted {len(self.person_names_blocklist)} person names")

        # Step 2: Discover fine-grained topics (15-20)
        print("\n" + "="*60)
        print(f"STEP 1: Fine-Grained Topic Discovery ({max_fine_topics} topics)")
        print("="*60)

        # Use smaller min_cluster_size to get more topics
        fine_centroids, fine_topics = self.cluster_embeddings_bertopic(
            min_cluster_size=20,  # Smaller = more topics
            use_kmeans=True,      # K-means for consistent results
            n_clusters=max_fine_topics
        )

        # Name fine-grained topics
        print("\n" + "="*60)
        print("Naming Fine-Grained Topics")
        print("="*60)

        fine_category_mapping = {}
        used_names = []

        for topic_id in sorted(fine_centroids.keys()):
            bertopic_keywords = self.get_bertopic_keywords(topic_id, top_n=10)

            cluster_analysis = {
                "cluster_id": topic_id,
                "size": len([c for c in self.email_chunks
                           if c.meta.get("cluster_id") == topic_id]),
                "top_keywords": bertopic_keywords,
                "sample_subjects": [c.meta.get("subject", "")[:100]
                                   for c in self.email_chunks
                                   if c.meta.get("cluster_id") == topic_id][:5],
                "common_body_words": bertopic_keywords[:5]
            }

            name = self._llm_name_cluster(cluster_analysis, used_names)
            fine_category_mapping[topic_id] = name
            used_names.append(name)
            print(f"  Topic {topic_id:2d}: '{name}' ({cluster_analysis['size']} emails)")

        # Step 3: Cluster topic centroids to create broad categories
        print("\n" + "="*60)
        print(f"STEP 2: Hierarchical Clustering → {n_broad_categories} Broad Categories")
        print("="*60)

        from sklearn.cluster import AgglomerativeClustering

        # Prepare centroid matrix
        topic_ids = sorted(fine_centroids.keys())
        topic_centroids = np.array([fine_centroids[tid] for tid in topic_ids])

        # Hierarchical clustering on topics
        hierarchical = AgglomerativeClustering(
            n_clusters=n_broad_categories,
            linkage='ward'
        )
        broad_assignments = hierarchical.fit_predict(topic_centroids)

        # Build hierarchy: broad_id → [specific topic_ids]
        hierarchy = {}
        for topic_id, broad_id in zip(topic_ids, broad_assignments):
            if broad_id not in hierarchy:
                hierarchy[broad_id] = []
            hierarchy[broad_id].append(topic_id)

        print(f"✓ Created {len(hierarchy)} broad categories")
        for broad_id, specific_ids in sorted(hierarchy.items()):
            print(f"  Broad {broad_id}: {len(specific_ids)} specific topics → {specific_ids}")

        # Step 4: Name broad categories
        print("\n" + "="*60)
        print("STEP 3: Naming Broad Categories")
        print("="*60)

        broad_category_mapping = {}
        broad_used_names = []

        for broad_id, specific_ids in sorted(hierarchy.items()):
            # Get all emails in this broad category
            broad_emails = [e for e in self.email_chunks
                           if e.meta.get("cluster_id") in specific_ids]

            # Collect representative keywords from all specific topics
            all_keywords = []
            for specific_id in specific_ids:
                all_keywords.extend(self.get_bertopic_keywords(specific_id, top_n=5))

            # Get most common keywords (top 10)
            from collections import Counter
            keyword_counts = Counter(all_keywords)
            top_keywords = [kw for kw, _ in keyword_counts.most_common(10)]

            # Sample subjects from across all specific topics
            sample_subjects = [e.meta.get("subject", "")[:100]
                             for e in broad_emails[:10]]

            # Use LLM to name broad category
            cluster_analysis = {
                "cluster_id": broad_id,
                "size": len(broad_emails),
                "top_keywords": top_keywords,
                "sample_subjects": sample_subjects,
                "common_body_words": top_keywords[:5],
                "subcategories": [fine_category_mapping[sid] for sid in specific_ids]
            }

            # Modified prompt for broad categories
            broad_name = self._llm_name_broad_cluster(cluster_analysis, broad_used_names)
            broad_category_mapping[broad_id] = broad_name
            broad_used_names.append(broad_name)

            print(f"\n  Broad Category {broad_id}: '{broad_name}'")
            print(f"    Size: {len(broad_emails)} emails")
            print(f"    Subcategories ({len(specific_ids)}):")
            for sid in specific_ids:
                specific_name = fine_category_mapping[sid]
                specific_count = len([e for e in self.email_chunks
                                     if e.meta.get("cluster_id") == sid])
                print(f"      - {specific_name} ({specific_count} emails)")

        # Step 5: Build complete hierarchy structure
        hierarchy_structure = {
            "broad_categories": {
                broad_id: {
                    "name": broad_category_mapping[broad_id],
                    "specific_topics": [
                        {
                            "id": sid,
                            "name": fine_category_mapping[sid],
                            "email_count": len([e for e in self.email_chunks
                                              if e.meta.get("cluster_id") == sid]),
                            "keywords": self.get_bertopic_keywords(sid, top_n=5)
                        }
                        for sid in specific_ids
                    ],
                    "total_emails": len([e for e in self.email_chunks
                                        if e.meta.get("cluster_id") in specific_ids])
                }
                for broad_id, specific_ids in hierarchy.items()
            },
            "metadata": {
                "n_broad_categories": n_broad_categories,
                "n_specific_topics": len(fine_centroids),
                "total_emails": len(self.email_chunks),
                "creation_method": "hierarchical_agglomerative"
            }
        }

        # Step 6: Save results
        if output_path is None:
            output_path = Path("data/categories/hierarchical_categories.json")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(hierarchy_structure, f, indent=2, ensure_ascii=False)

        print("\n" + "="*60)
        print("HIERARCHY SUMMARY")
        print("="*60)
        print(f"Broad categories: {n_broad_categories}")
        print(f"Specific topics: {len(fine_centroids)}")
        print(f"Total emails: {len(self.email_chunks)}")
        print(f"Saved to: {output_path}")
        print("="*60)

        print("\n✅ Hierarchical category discovery complete!")

        return hierarchy_structure

    def _llm_name_broad_cluster(self, cluster_analysis: Dict, used_names: List[str]) -> str:
        """Name a broad category using LLM (modified prompt for higher-level categories)."""
        subcategories_str = ", ".join(cluster_analysis.get("subcategories", []))
        keywords_str = ", ".join(cluster_analysis["top_keywords"][:10])
        subjects_str = "\n".join([f"- {s}" for s in cluster_analysis["sample_subjects"][:5]])

        used_names_str = ", ".join(used_names) if used_names else "None"

        prompt = f"""You are naming a BROAD email category that groups several related subcategories.

SUBCATEGORIES IN THIS BROAD CATEGORY:
{subcategories_str}

REPRESENTATIVE KEYWORDS:
{keywords_str}

SAMPLE EMAIL SUBJECTS:
{subjects_str}

SIZE: {cluster_analysis['size']} emails

TASK: Create a SHORT, CLEAR, BROAD category name (2-4 words) that encompasses all subcategories.

REQUIREMENTS:
- Must be broad enough to cover all subcategories
- 2-4 words maximum
- Title Case
- Use generic, high-level terms
- Must NOT be similar to already used names: {used_names_str}

GOOD EXAMPLES:
- "Technical Issues"
- "Product Information"
- "User Support"
- "System Configuration"
- "Research Assistance"

BAD EXAMPLES:
- "User Interface Customization Options for Research Portal" (too specific and long)
- "Emails" (too generic)
- "Miscellaneous" (not descriptive)

Return ONLY the category name, nothing else."""

        try:
            completer = OpenAICompleter()
            response = completer.complete(
                prompt=prompt,
                temperature=0.3,
                max_tokens=20
            )

            name = response.strip().strip('"').strip("'")

            # Fallback if LLM fails
            if not name or len(name.split()) > 5:
                subcats = cluster_analysis.get("subcategories", [])
                name = f"Broad Category {cluster_analysis['cluster_id']}"
                if subcats:
                    # Extract common words from subcategories
                    common_words = set(subcats[0].split())
                    for subcat in subcats[1:]:
                        common_words &= set(subcat.split())
                    if common_words:
                        name = " ".join(list(common_words)[:3])

            return name

        except Exception as e:
            print(f"    ⚠️  LLM naming failed: {e}")
            return f"Broad Category {cluster_analysis['cluster_id']}"


def main():
    import argparse

    parser = argparse.ArgumentParser(description="BERTopic-based category discovery")
    parser.add_argument("--project", type=str,
                       default="data/projects/Primo_List")
    parser.add_argument("--min-cluster-size", type=int, default=30,
                       help="Minimum emails per topic (HDBSCAN only)")
    parser.add_argument("--use-kmeans", action="store_true",
                       help="Use K-means instead of HDBSCAN (no outliers)")
    parser.add_argument("--n-clusters", type=int, default=5,
                       help="Number of clusters (K-means only)")
    parser.add_argument("--output", type=str,
                       default="data/categories/discovered_categories_bertopic.json")
    parser.add_argument("--auto", action="store_true",
                       help="Non-interactive mode")

    args = parser.parse_args()

    print("="*60)
    print("  BERTOPIC EMAIL CATEGORY DISCOVERY")
    print("="*60)
    print(f"\nProject: {args.project}")
    if args.use_kmeans:
        print(f"Algorithm: K-means (n_clusters={args.n_clusters})")
    else:
        print(f"Algorithm: HDBSCAN (min_cluster_size={args.min_cluster_size})")
    print(f"Output: {args.output}")

    project = ProjectManager(Path(args.project))
    discovery = BERTopicCategoryDiscovery(project)
    discovery.run_bertopic(
        min_cluster_size=args.min_cluster_size,
        output_path=Path(args.output),
        interactive=not args.auto,
        use_kmeans=args.use_kmeans,
        n_clusters=args.n_clusters
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
