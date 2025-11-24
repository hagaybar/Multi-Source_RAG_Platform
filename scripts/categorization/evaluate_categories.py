#!/usr/bin/env python3
"""
Evaluate discovered email categories using multiple metrics.

Metrics:
- Silhouette Score (cluster quality)
- Topic Coherence (keyword meaningfulness)
- Category Distribution (balance)
- Sample Analysis (qualitative)
"""

import sys
import json
import random
import numpy as np
from pathlib import Path
from collections import Counter
from typing import List, Dict, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.core.project_manager import ProjectManager
from scripts.retrieval.base import FaissRetriever


def load_categories(categories_path: Path) -> Dict:
    """Load discovered categories from JSON."""
    with open(categories_path, 'r') as f:
        return json.load(f)


def load_emails_with_embeddings(project: ProjectManager) -> Tuple[List[Dict], np.ndarray, List[int]]:
    """Load emails with embeddings and cluster assignments.

    Returns:
        emails: List of email metadata
        embeddings: Array of embeddings
        cluster_labels: List of cluster IDs
    """
    # Load metadata
    metadata_path = project.get_metadata_path("outlook_eml")
    emails = []

    with open(metadata_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                emails.append(json.loads(line))

    # Load embeddings from FAISS
    index_path = project.root_dir / "output" / "faiss" / "outlook_eml.faiss"
    retriever = FaissRetriever(index_path, metadata_path)

    embeddings_list = []
    for i in range(len(emails)):
        embedding = retriever.index.reconstruct(i)
        embeddings_list.append(embedding)

    embeddings = np.array(embeddings_list)

    return emails, embeddings, None


def assign_clusters_from_centroids(embeddings: np.ndarray, centroids: np.ndarray) -> List[int]:
    """Assign emails to nearest centroid from discovered categories.

    Args:
        embeddings: Email embeddings (n_emails, embedding_dim)
        centroids: Cluster centroids from discovery (n_clusters, embedding_dim)

    Returns:
        List of cluster IDs (one per email)
    """
    from scipy.spatial.distance import cdist

    # Compute distance from each email to each centroid
    distances = cdist(embeddings, centroids, metric='euclidean')

    # Assign to nearest centroid
    cluster_labels = np.argmin(distances, axis=1)

    return cluster_labels.tolist()


def compute_silhouette_score(embeddings: np.ndarray, cluster_labels: List[int]) -> float:
    """Compute silhouette score for clustering quality."""
    from sklearn.metrics import silhouette_score

    if len(set(cluster_labels)) < 2:
        return 0.0

    score = silhouette_score(embeddings, cluster_labels, metric='euclidean')
    return score


def compute_topic_coherence(category_keywords: List[str], emails_in_category: List[Dict]) -> float:
    """Compute topic coherence based on keyword co-occurrence.

    Simplified coherence: measures how often keywords appear together.
    """
    if not category_keywords or not emails_in_category:
        return 0.0

    # Count co-occurrences
    co_occurrence = 0
    total_checks = 0

    for email in emails_in_category:
        text = (email.get('subject', '') + ' ' + email.get('text', '')).lower()

        for i, word1 in enumerate(category_keywords):
            for word2 in category_keywords[i+1:]:
                total_checks += 1
                if word1 in text and word2 in text:
                    co_occurrence += 1

    if total_checks == 0:
        return 0.0

    coherence = co_occurrence / total_checks
    return coherence


def evaluate_category_balance(cluster_labels: List[int]) -> Dict:
    """Evaluate category distribution balance."""
    counts = Counter(cluster_labels)
    total = len(cluster_labels)

    sizes = list(counts.values())
    avg_size = np.mean(sizes)
    std_size = np.std(sizes)
    min_size = min(sizes)
    max_size = max(sizes)

    # Coefficient of variation (lower is more balanced)
    cv = (std_size / avg_size) if avg_size > 0 else 0

    return {
        'total_emails': total,
        'n_categories': len(counts),
        'avg_size': avg_size,
        'std_size': std_size,
        'min_size': min_size,
        'max_size': max_size,
        'coefficient_variation': cv,
        'balance_score': 1 - min(cv, 1.0)  # 1 = perfectly balanced, 0 = very imbalanced
    }


def sample_emails_per_category(emails: List[Dict], cluster_labels: List[int],
                                categories: Dict, n_samples: int = 3) -> Dict:
    """Sample random emails from each category for qualitative analysis."""
    category_mapping = categories['category_mapping']
    samples = {}

    # Group emails by cluster
    clusters = {}
    for email, label in zip(emails, cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(email)

    # Sample from each cluster
    for cluster_id, cluster_emails in clusters.items():
        category_name = category_mapping.get(str(cluster_id), 'Unknown')

        sampled = random.sample(cluster_emails, min(n_samples, len(cluster_emails)))

        samples[category_name] = [
            {
                'subject': email.get('subject', 'No subject')[:100],
                'text': email.get('text', '')[:200],
                'sender': email.get('sender_name', 'Unknown')[:40],
                'date': email.get('date', '')[:10]
            }
            for email in sampled
        ]

    return samples


def main():
    print("="*80)
    print("EMAIL CATEGORIZATION EVALUATION")
    print("="*80)

    # Paths
    project_path = Path("data/projects/Primo_List_2")
    categories_path = Path("data/categories/discovered_categories_bertopic_kmeans.json")

    # Load data
    print("\n1. Loading data...")
    project = ProjectManager(project_path)
    categories = load_categories(categories_path)
    emails, embeddings, _ = load_emails_with_embeddings(project)

    # Load centroids from Phase 1 discovery and assign clusters
    # Centroids are stored as dict: {category_name: embedding_list}
    # Need to convert to 2D array ordered by cluster_id
    centroids_dict = categories['centroids']
    category_mapping = categories['category_mapping']

    # Create ordered list of centroids by cluster_id
    centroids_list = []
    for cluster_id in sorted(category_mapping.keys(), key=int):
        category_name = category_mapping[cluster_id]
        centroids_list.append(centroids_dict[category_name])

    centroids = np.array(centroids_list)
    cluster_labels = assign_clusters_from_centroids(embeddings, centroids)

    print(f"   ✓ Loaded {len(emails)} emails")
    print(f"   ✓ Loaded {len(embeddings)} embeddings")
    print(f"   ✓ Loaded {len(centroids)} category centroids")
    print(f"   ✓ Assigned {len(set(cluster_labels))} clusters")

    # Metric 1: Silhouette Score
    print("\n2. Computing Silhouette Score (cluster quality)...")
    silhouette = compute_silhouette_score(embeddings, cluster_labels)
    print(f"   Silhouette Score: {silhouette:.4f}")
    print(f"   Interpretation: ", end="")
    if silhouette > 0.5:
        print("Excellent - Strong, well-separated clusters")
    elif silhouette > 0.25:
        print("Good - Reasonable cluster structure")
    elif silhouette > 0:
        print("Fair - Weak cluster structure, some overlap")
    else:
        print("Poor - Incorrect clustering")

    # Metric 2: Category Balance
    print("\n3. Evaluating Category Balance...")
    balance = evaluate_category_balance(cluster_labels)
    print(f"   Total emails: {balance['total_emails']}")
    print(f"   Categories: {balance['n_categories']}")
    print(f"   Average size: {balance['avg_size']:.1f} emails")
    print(f"   Std deviation: {balance['std_size']:.1f}")
    print(f"   Range: {balance['min_size']} - {balance['max_size']} emails")
    print(f"   Balance score: {balance['balance_score']:.3f} (1.0 = perfect)")

    # Metric 3: Topic Coherence (for categories with keywords)
    print("\n4. Computing Topic Coherence (keyword meaningfulness)...")

    # Group emails by cluster
    clusters = {}
    for email, label in zip(emails, cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(email)

    coherence_scores = {}
    for cluster_id, category_name in categories['category_mapping'].items():
        rules = categories['rules'].get(category_name, {})
        keywords = rules.get('subject_keywords', []) + rules.get('body_keywords', [])

        cluster_emails = clusters.get(int(cluster_id), [])

        if keywords and cluster_emails:
            coherence = compute_topic_coherence(keywords[:5], cluster_emails[:50])
            coherence_scores[category_name] = coherence
            print(f"   {category_name}: {coherence:.4f}")

    avg_coherence = np.mean(list(coherence_scores.values())) if coherence_scores else 0
    print(f"\n   Average Coherence: {avg_coherence:.4f}")

    # Metric 4: Sample Analysis
    print("\n5. Sampling emails for qualitative analysis...")
    samples = sample_emails_per_category(emails, cluster_labels, categories, n_samples=2)

    print("\n" + "="*80)
    print("SAMPLE EMAILS PER CATEGORY")
    print("="*80)

    for category_name, category_samples in samples.items():
        print(f"\n### {category_name.upper()}")
        print("-"*80)

        for i, sample in enumerate(category_samples, 1):
            print(f"\n  [{i}] {sample['date']} - {sample['sender']}")
            print(f"      Subject: {sample['subject']}")
            print(f"      Preview: {sample['text'][:120]}...")

    # Summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"\n✓ Silhouette Score: {silhouette:.4f} ", end="")
    print("✅ Good" if silhouette > 0.25 else "⚠️ Needs improvement")

    print(f"✓ Balance Score: {balance['balance_score']:.3f} ", end="")
    print("✅ Balanced" if balance['balance_score'] > 0.7 else "⚠️ Imbalanced")

    print(f"✓ Topic Coherence: {avg_coherence:.4f} ", end="")
    print("✅ Good" if avg_coherence > 0.3 else "⚠️ Needs improvement")

    # Overall quality
    overall_score = (silhouette * 0.4 + balance['balance_score'] * 0.3 + avg_coherence * 0.3)
    print(f"\n✓ Overall Quality Score: {overall_score:.3f} / 1.0")

    if overall_score > 0.6:
        print("   🎉 Excellent categorization quality!")
    elif overall_score > 0.4:
        print("   👍 Good categorization, minor improvements possible")
    else:
        print("   ⚠️ Needs improvement - consider adjusting parameters")

    print("\n" + "="*80)


if __name__ == "__main__":
    sys.exit(main())
