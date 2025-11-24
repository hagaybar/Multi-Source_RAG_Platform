# Email Categorization System - Operational Plan

**Date:** 2025-11-22 (Updated with Phase 1 Completion)
**Priority:** HIGH
**Status:** ✅ Phase 1 Complete → Phase 2 Ready
**Current Quality:** 0.371 / 1.0 (Phase 1 - Improved +46%)
**Target Quality:** >0.60 / 1.0 (Production-Ready)
**Implementation:** 3-Phase Approach (Quick Wins → BERTopic → Advanced)

**Related Documents:**
- Evaluation Report: `docs/future/EMAIL_CATEGORIZATION_EVALUATION.md`
- Phase 1 Results: `docs/future/EMAIL_CATEGORIZATION_PHASE1_RESULTS.md` ✅ NEW
- Test Results: `data/categories/discovered_categories_phase1.json`
- Evaluation Script: `scripts/categorization/evaluate_categories.py`

---

## 🎯 EXECUTIVE SUMMARY

**Problem:** Similarity-based retrieval fails for aggregation queries ("What topics were discussed recently?")

**Solution:** Ingestion-time categorization with 3-tier confidence-based system

**Current Status (Phase 1 Complete):**
- ✅ Discovery script implemented (K-means + LLM naming)
- ✅ TF-IDF keyword extraction (+115% coherence improvement!)
- ✅ System artifact filtering (reactions, auto-replies)
- ✅ Evaluation framework created
- ✅ **Quality Score: 0.371 / 1.0** (+46% improvement from 0.254)
- ⚠️ K-means clustering confirmed as bottleneck

**Phase 1 Results:**
| Metric | Baseline | Phase 1 | Change | Status |
|--------|----------|---------|--------|--------|
| Silhouette Score | 0.026 | 0.0195 | -25% | ⚠️ K-means limitation |
| Balance Score | 0.557 | 0.661 | +19% | ⚠️ Improved |
| Topic Coherence | 0.256 | 0.5492 | **+115%** | ✅ **Exceeded target!** |
| Overall Quality | 0.254 | 0.371 | **+46%** | ⚠️ Progress made |

**Key Achievements:**
1. ✅ TF-IDF solves keyword quality problem (coherence exceeds target)
2. ✅ Bigram extraction captures meaningful phrases
3. ✅ Meaningful categories: "Research Support", "UI Customization", etc.
4. ⚠️ Clustering algorithm bottleneck confirmed (need BERTopic)

**Path Forward:** Phase 2 (BERTopic) → Phase 3 (Advanced) reaching 0.68 quality score

---

## ✅ PHASE 1 COMPLETION STATUS (2025-11-22)

**Status:** COMPLETE ✅
**Quality Improvement:** 0.254 → 0.371 (+46%)
**Duration:** ~5 minutes execution time
**Files Modified:** 2 scripts, 1 results file created

### Accomplishments

#### ✅ Task 1.1: Reduced Categories (7 → 5)
- Better balance: 145-379 emails per category (vs 45-301 baseline)
- Less overlap between categories
- Balance score: 0.557 → 0.661 (+19%)

#### ✅ Task 1.2: TF-IDF Keyword Extraction
- **MAJOR SUCCESS:** Coherence: 0.256 → 0.5492 (+115%)
- **Exceeded target** (0.40) by 37%!
- Successfully captures bigrams ("research assistant", "guardrails in")
- 480-word stoplist filters noise (names, artifacts)

#### ✅ Task 1.3: System Artifact Filtering
- Filtered 20 emails (1.7%): reactions, auto-replies, short messages
- Cleaner dataset for clustering

### Phase 1 Final Metrics

| Metric | Baseline | Phase 1 | Change | Target | Status |
|--------|----------|---------|--------|--------|--------|
| **Silhouette** | 0.026 | 0.0195 | -25% | 0.15 | ⚠️ K-means bottleneck |
| **Balance** | 0.557 | 0.661 | +19% | 0.70 | ⚠️ Close |
| **Coherence** | 0.256 | 0.5492 | **+115%** | 0.40 | ✅ **Exceeded!** |
| **Overall** | 0.254 | 0.371 | **+46%** | 0.42 | ⚠️ Close |

### Key Findings

**✅ What Worked:**
- TF-IDF keyword extraction is a **game-changer** (+115% coherence)
- Bigram support captures meaningful phrases
- LLM category naming produces interpretable labels
- System artifact filtering removes noise effectively

**⚠️ What Didn't:**
- Silhouette score actually declined (-25%)
- Confirms K-means is wrong algorithm (as research predicted)
- Overall quality improved but missed ambitious +65% target

**🎯 Conclusion:**
Keyword extraction problem is **SOLVED** ✅. Clustering algorithm is the **BOTTLENECK** ⚠️. Phase 2 (BERTopic) is essential and should maintain excellent keywords while dramatically improving cluster structure.

**Detailed Analysis:** See `docs/future/EMAIL_CATEGORIZATION_PHASE1_RESULTS.md`

---

## 🎯 Problem Statement

**Current Limitation:**
RAG systems use similarity-based retrieval, which doesn't work well for aggregation queries like:
> "I need to learn about the recent topics that were discussed in the list during the last couple of weeks"

**What Users Want:**
- Comprehensive topic overview (ALL recent topics, not just top-k similar chunks)
- Categorical organization (grouped by topic/theme)
- Summarized discussions for each topic

**What RAG Currently Does:**
- Embeds query: "learn about recent topics"
- Finds top-k chunks most similar to query
- Summarizes only those chunks (misses many topics)

**The Gap:** Similarity search ≠ Comprehensive aggregation

---

## 💡 Proposed Solution: Ingestion-Time Categorization

Instead of analyzing emails at query time, **categorize them once during ingestion** and store categories as metadata.

### Key Advantages:

1. **Pay Once, Use Forever**
   - Categorize each email once during ingestion (~$0.001 per email)
   - Every aggregation query benefits instantly (no re-processing)
   - Query time: Just group by metadata (instant!)

2. **Better Query Experience**
   ```python
   # Query time (FAST):
   recent_emails = get_all_recent(days_back=14)
   by_category = group_by(recent_emails, "category")  # Metadata grouping
   # Then summarize each category
   ```

3. **Enables New Features**
   - UI filters: "Show me all Bug Reports"
   - Analytics: "How many feature requests this month?"
   - Trends: "Which topics are increasing?"

---

## 🏗️ Architecture: 3-Tier Categorization System

### Design Principle: Confidence-Based Selection

**NOT a waterfall** (Tier 1 → Tier 2 → Tier 3)
**BUT confidence voting** (all tiers run, highest confidence wins)

```python
results = [
    ("rule", "Bug Reports", confidence=0.65),
    ("embedding", "Questions", confidence=0.89),  # Winner!
    ("llm", "Bug Reports", confidence=0.95)       # Not called (0.89 > threshold)
]

final_category = max(results, key=lambda r: r.confidence)
```

---

### Tier 1: Rule-Based (Free, Fast, 70% coverage)

**How it works:**
- Pattern matching on subject lines and body text
- Keyword-based heuristics (e.g., "bug", "error" → Bug Reports)
- Rules discovered from data analysis (not hard-coded guesses)

**Confidence:** Fixed at 0.65 (medium confidence)

**Example:**
```python
def tier1_rules(email):
    subject = email.meta["subject"].lower()

    if any(word in subject for word in ["bug", "error", "issue", "broken"]):
        return ("Bug Reports", 0.65)

    if any(word in subject for word in ["feature", "request", "enhancement"]):
        return ("Feature Requests", 0.65)

    # ... more rules

    return None  # No match
```

**Cost:** $0
**Speed:** <1ms per email

---

### Tier 2: Embedding Similarity (Cheap, Fast, 20% coverage)

**How it works:**
- Compare email embedding to category centroids
- Cosine similarity gives confidence score
- Only used when rule confidence is low or no rule matches

**Confidence:** Variable (0.0 - 1.0 based on similarity)

**Example:**
```python
def tier2_embedding(email):
    email_emb = email.embedding  # Already exists from RAG!

    similarities = {}
    for category, centroid in category_centroids.items():
        sim = cosine_similarity(email_emb, centroid)
        similarities[category] = sim

    best_category = max(similarities, key=similarities.get)
    best_score = similarities[best_category]

    return (best_category, best_score)  # Score IS the confidence
```

**Cost:** $0 (reuses existing embeddings)
**Speed:** ~10ms per email

---

### Tier 3: LLM Classification (Fallback, 10% coverage)

**How it works:**
- GPT-3.5-turbo categorizes ambiguous emails
- Only called when Tier 1 + Tier 2 confidence < 0.85
- Or periodically for validation (sampling)

**Confidence:** Fixed at 0.95 (high confidence)

**Example:**
```python
def tier3_llm(email):
    prompt = f"""Categorize this email into ONE category:
    - Bug Reports
    - Feature Requests
    - Questions
    - Announcements
    - Configuration
    - Discussion
    - Other

    Subject: {email.subject}
    Body: {email.text[:500]}

    Return ONLY the category name."""

    response = gpt35_turbo(prompt)
    return (response.strip(), 0.95)
```

**Cost:** ~$0.001 per email (only for 10% of emails)
**Speed:** ~500ms per email

---

## 🥾 Bootstrap Process: Data-Driven Discovery

**Key Insight:** You have 6 months of emails with embeddings already computed!

### Phase 1: Discover Categories from Existing Data

**Input:**
- 6 months of raw email data
- All embeddings (already computed for RAG)
- Email metadata (subjects, senders, dates)

**Process:**

1. **Cluster Embeddings** (K-means or HDBSCAN)
   ```python
   # Auto-discover natural groupings
   embeddings = [chunk.embedding for chunk in all_emails]
   clusters = KMeans(n_clusters=7).fit(embeddings)
   ```

2. **Analyze Clusters** (Inductive)
   - Most common subject keywords
   - Sample email subjects
   - Sender patterns
   - Temporal patterns

3. **Name Categories** (Manual)
   - Based on cluster analysis
   - E.g., Cluster 0 → "Bug Reports" (keywords: error, issue, bug)

4. **Extract Rules** (Automatic)
   - Keywords that appear in >30% of emails in cluster
   - These become Tier 1 rules

5. **Compute Centroids** (Automatic)
   - Mean embedding of all emails in category
   - These become Tier 2 centroids

**Output:**
- 7 discovered categories (data-driven, not pre-defined)
- Rule patterns for Tier 1
- Real centroids for Tier 2
- Ready to categorize new emails!

**Cost:** $0 (all data exists)
**Time:** ~5 minutes of computation + 30 minutes of analysis

---

### Phase 2: Categorize New Incoming Emails

**For each new email during ingestion:**

```python
def categorize_email(email):
    results = []

    # Tier 1: Try rules
    rule_result = tier1_rules(email)
    if rule_result:
        results.append(("rule", rule_result[0], rule_result[1]))

    # Tier 2: Try embedding similarity
    emb_result = tier2_embedding(email)
    results.append(("embedding", emb_result[0], emb_result[1]))

    # Tier 3: LLM fallback (only if confidence < 0.85)
    max_conf = max(r[2] for r in results)
    if max_conf < 0.85:
        llm_result = tier3_llm(email)
        results.append(("llm", llm_result[0], llm_result[1]))

        # Update centroid with LLM result (ground truth)
        update_category_centroid(llm_result[0], email.embedding)

    # Pick highest confidence
    best_source, best_category, best_conf = max(results, key=lambda r: r[2])

    # Store in metadata
    email.meta["category"] = best_category
    email.meta["category_source"] = best_source
    email.meta["category_confidence"] = best_conf

    return best_category
```

**Expected Distribution:**
- 70%: Tier 1 (rules) - $0
- 20%: Tier 2 (embeddings) - $0
- 10%: Tier 3 (LLM) - ~$0.001 each

**Cost for 1000 emails:** ~$0.10 - $0.15

---

### Phase 3: Continuous Improvement

**Centroid Refinement:**
```python
def update_category_centroid(category, new_embedding):
    """Moving average: centroids improve over time."""
    old_centroid = category_centroids[category]
    count = category_counts[category]

    new_centroid = (old_centroid * count + new_embedding) / (count + 1)

    category_centroids[category] = new_centroid
    category_counts[category] = count + 1
```

**Validation Sampling:**
- Every 100 emails: Sample 10 random emails
- Compare embedding categorization vs LLM categorization
- If accuracy < 85%, trigger retraining

**Category Evolution:**
- If 20+ emails land in "Other" with similar content
- LLM suggests new category: "Performance Issues"
- User approves → new category added

---

## 📊 Integration with Aggregation Queries

**Before (Similarity-Based):**
```python
# User query: "What topics were discussed last 2 weeks?"
query_emb = embed("topics discussed")
chunks = retrieve_top_k(query_emb, k=20)  # Misses many topics!
summary = llm.summarize(chunks)
```

**After (Category-Based):**
```python
# User query: "What topics were discussed last 2 weeks?"
intent = detect_intent(query)  # "aggregation_query"
temporal = extract_temporal(query)  # days_back=14

# Get ALL recent emails
recent = get_all_recent(days_back=14)

# Group by category (instant!)
by_category = {}
for chunk in recent:
    cat = chunk.meta["category"]
    by_category.setdefault(cat, []).append(chunk)

# For each category, get representative samples
summaries = {}
for category, chunks in by_category.items():
    representative = chunks[:5]  # Most recent
    summaries[category] = representative

# Single LLM call for structured summary
prompt = f"""
You have {len(recent)} emails from last 14 days, organized into {len(by_category)} categories:

Category: Bug Reports ({len(by_category['Bug Reports'])} emails)
Sample emails:
- [subject 1]
- [subject 2]

Category: Feature Requests ({len(by_category['Feature Requests'])} emails)
Sample emails:
- [subject 1]
- [subject 2]

...

Please provide:
1. Brief summary of each category
2. Key points discussed
3. Notable trends
"""

summary = llm.generate(prompt)
```

**Benefits:**
- ✅ Comprehensive (ALL topics covered)
- ✅ Organized (grouped by category)
- ✅ Fast (no re-analysis at query time)
- ✅ Accurate (based on ingestion-time categorization)

---

## 🎯 Expected Category Schema (Discovered from Data)

**Initial categories** (will be discovered, not pre-defined):

1. **Bug Reports** - Technical issues, errors, problems
2. **Feature Requests** - New features, enhancements
3. **Questions** - How-to, clarifications, help needed
4. **Announcements** - Release notes, updates, news
5. **Configuration** - Setup, deployment, settings
6. **Discussion** - General conversation, opinions
7. **Other** - Everything else

**Categories evolve organically:**
- Discovered from actual email patterns
- Can be split/merged based on usage
- New categories added when patterns emerge

---

## 📈 Success Metrics

**Categorization Accuracy:**
- Target: >85% agreement with human labeling
- Measured by: Sampling 50 emails/week, manual validation

**Query Performance:**
- Aggregation queries: <2 seconds (down from >30 seconds)
- Cost per aggregation query: $0.01 (down from $0.05)

**Coverage:**
- Tier 1 (rules): 70% of emails
- Tier 2 (embeddings): 20% of emails
- Tier 3 (LLM): 10% of emails

**Cost:**
- Initial discovery: $0 (uses existing data)
- Per-email categorization: ~$0.0001 (averaged)
- Re-categorization of 6 months: ~$100 if using LLM for all (optional)

---

## 🚀 OPERATIONAL IMPLEMENTATION PLAN

### Completed Baseline Work ✅

- [x] Discovery script implemented (`category_discovery.py`)
- [x] LLM-based cluster naming functional
- [x] Comprehensive keyword filtering (stopwords + person names + patterns)
- [x] Evaluation framework created (`evaluate_categories.py`)
- [x] Initial baseline metrics established (Quality: 0.254)
- [x] Research completed on BERTopic, TF-IDF, and topic modeling best practices
- [x] Root cause analysis completed

---

## 📋 PHASE 1: QUICK WINS (2-3 Hours) → Target Quality: 0.42

**Goal:** Improve current system without changing architecture (+65% quality improvement)

**Expected Results:**
- Silhouette: 0.026 → 0.15 (+480%)
- Balance: 0.557 → 0.70 (+26%)
- Coherence: 0.256 → 0.40 (+56%)
- **Overall: 0.254 → 0.42 (+65%)**

---

### Task 1.1: Merge Overlapping Categories ⏱️ 30 min

**Problem:** "User Interface Issues" vs "Customization" are too similar, causing cluster overlap

**Action:**
```bash
# Re-run discovery with fewer categories
PYTHONPATH=. poetry run python scripts/categorization/category_discovery.py \
    --project data/projects/Primo_List \
    --n-categories 5 \
    --output data/categories/discovered_categories_5cats.json \
    --auto
```

**Proposed Category Consolidation:**
```
Before (7 categories):                    After (5 categories):
- User Interface Issues          ─┐
- User Interface Customization   ─┴→ User Interface
- Technical Support              ─┐
- Technical Problems             ─┴→ Technical Issues
- Product Updates                  → Product Updates
- Research Assistant Inquiries     → Research Assistant
- Email Management                 → (filter as noise)
```

**Success Criteria:**
- ✅ Silhouette score increases (less overlap)
- ✅ Balance score improves (larger clusters)
- ✅ No category has <50 emails

**Files Modified:**
- None (just re-run with different parameter)

---

### Task 1.2: Implement TF-IDF Keyword Extraction ⏱️ 1 hour

**Problem:** Simple frequency counting misses distinctive keywords

**Action:** Replace frequency-based extraction with TF-IDF

**File:** `scripts/categorization/category_discovery.py`

**Code Changes:**

```python
# ADD after imports
from sklearn.feature_extraction.text import TfidfVectorizer

# MODIFY extract_rules() method
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

        # TF-IDF for subject keywords
        if subject_texts:
            vectorizer_subj = TfidfVectorizer(
                max_features=10,
                ngram_range=(1, 2),  # Unigrams + bigrams
                stop_words=list(EMAIL_STOPWORDS),
                min_df=2  # Appear in at least 2 emails
            )

            try:
                tfidf_matrix = vectorizer_subj.fit_transform(subject_texts)
                feature_names = vectorizer_subj.get_feature_names_out()

                # Get average TF-IDF score per feature
                scores = tfidf_matrix.mean(axis=0).A1
                top_indices = scores.argsort()[-10:][::-1]
                subject_keywords = [feature_names[i] for i in top_indices]
            except:
                subject_keywords = []
        else:
            subject_keywords = []

        # TF-IDF for body keywords
        if body_texts:
            vectorizer_body = TfidfVectorizer(
                max_features=10,
                ngram_range=(1, 2),
                stop_words=list(EMAIL_STOPWORDS),
                min_df=2
            )

            try:
                tfidf_matrix = vectorizer_body.fit_transform(body_texts)
                feature_names = vectorizer_body.get_feature_names_out()

                scores = tfidf_matrix.mean(axis=0).A1
                top_indices = scores.argsort()[-10:][::-1]
                body_keywords = [feature_names[i] for i in top_indices]
            except:
                body_keywords = []
        else:
            body_keywords = []

        rules[category_name] = {
            "cluster_id": cluster_id,
            "subject_keywords": subject_keywords,
            "body_keywords": body_keywords,
            "confidence": 0.65,
            "sample_size": len(cluster_emails),
            "extraction_method": "tfidf"  # NEW: Track method
        }

        print(f"✓ {category_name}: {len(subject_keywords)} subject, "
              f"{len(body_keywords)} body keywords (TF-IDF)")

    return rules
```

**Success Criteria:**
- ✅ More subject_keywords extracted (currently many are empty)
- ✅ Keywords are distinctive (e.g., "research assistant" phrase, not just "search")
- ✅ Topic coherence score increases

**Validation:**
```bash
# Re-run evaluation after changes
PYTHONPATH=. poetry run python scripts/categorization/evaluate_categories.py
```

---

### Task 1.3: Filter Email System Artifacts ⏱️ 30 min

**Problem:** Email reaction messages pollute data (e.g., "<outlook-cdn.../reaction/like.png>")

**File:** `scripts/categorization/category_discovery.py`

**Code Changes:**

```python
# ADD after _extract_keywords() method
def _is_system_artifact(self, chunk: Chunk) -> bool:
    """Detect and filter email system artifacts.

    Examples:
    - Outlook reaction notifications
    - Auto-replies
    - Email system messages
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

    # Auto-replies
    if 'out of office' in subject or 'automatic reply' in subject:
        return True

    # Very short emails (likely system messages)
    if len(chunk.text.strip()) < 50:
        return True

    return False

# MODIFY load_email_chunks() method
def load_email_chunks(self) -> List[Chunk]:
    """Load all email chunks with embeddings from project."""
    # ... existing code ...

    # Filter chunks (AFTER loading, BEFORE returning)
    chunks_with_emb = [c for c in chunks
                       if hasattr(c, 'embedding') and c.embedding is not None]

    # NEW: Filter system artifacts
    filtered_chunks = [c for c in chunks_with_emb
                      if not self._is_system_artifact(c)]

    removed = len(chunks_with_emb) - len(filtered_chunks)
    if removed > 0:
        print(f"✓ Filtered {removed} system artifacts")

    print(f"✓ Loaded embeddings for {len(filtered_chunks)} chunks")

    return filtered_chunks
```

**Success Criteria:**
- ✅ "Email Management" category disappears or shrinks significantly
- ✅ All metrics improve (cleaner data)

---

### Task 1.4: Lower Keyword Threshold ⏱️ 10 min

**Problem:** 30% threshold too strict after aggressive filtering

**File:** `scripts/categorization/category_discovery.py`

**Code Changes:**

```python
# IN extract_rules() method (if not using TF-IDF)
# CHANGE threshold from 0.3 to 0.15
threshold = len(cluster_emails) * 0.15  # Was 0.3
```

**Note:** If implementing Task 1.2 (TF-IDF), this change is not needed as TF-IDF handles frequency differently.

---

### Phase 1 Validation Checklist

**After completing all Phase 1 tasks:**

```bash
# 1. Re-run discovery with improvements
PYTHONPATH=. poetry run python scripts/categorization/category_discovery.py \
    --project data/projects/Primo_List \
    --n-categories 5 \
    --output data/categories/discovered_categories_phase1.json \
    --auto

# 2. Run evaluation
PYTHONPATH=. poetry run python scripts/categorization/evaluate_categories.py

# 3. Check results
```

**Expected Metrics:**
- [ ] Silhouette Score: ~0.15 (was 0.026)
- [ ] Balance Score: ~0.70 (was 0.557)
- [ ] Topic Coherence: ~0.40 (was 0.256)
- [ ] Overall Quality: ~0.42 (was 0.254)

**If metrics not met:** Debug and tune parameters before proceeding to Phase 2

---

## 📋 PHASE 2: BERTOPIC INTEGRATION (1-2 Days) → Target Quality: 0.58

**Goal:** Replace K-means with BERTopic for state-of-the-art topic modeling (+38% improvement)

**Expected Results:**
- Silhouette: 0.15 → 0.35 (+133%)
- Balance: 0.70 → 0.80 (+14%)
- Coherence: 0.40 → 0.60 (+50%)
- **Overall: 0.42 → 0.58 (+38%)**

**Research Justification:**
- BERTopic outperforms K-means for document/email clustering (ACM 2024)
- Auto-determines optimal number of topics (no guessing k=5 or k=7)
- Handles outliers via HDBSCAN (marks noise as topic -1)
- Uses c-TF-IDF (class-based, finds distinctive keywords automatically)
- Variable cluster sizes (realistic for mailing lists)

---

### Task 2.1: Install BERTopic Dependencies ⏱️ 15 min

**Action:**

```bash
# Add to project dependencies
poetry add bertopic umap-learn hdbscan scikit-learn

# Verify installation
poetry run python -c "import bertopic; print(f'BERTopic {bertopic.__version__}')"
poetry run python -c "import umap; print(f'UMAP {umap.__version__}')"
poetry run python -c "import hdbscan; print(f'HDBSCAN {hdbscan.__version__}')"
```

**Success Criteria:**
- ✅ All imports successful
- ✅ No version conflicts

---

### Task 2.2: Create BERTopic Discovery Script ⏱️ 3-4 hours

**New File:** `scripts/categorization/bertopic_discovery.py`

**Implementation:**

```python
#!/usr/bin/env python3
"""
BERTopic-based Email Category Discovery

Uses state-of-the-art topic modeling instead of K-means.
Automatically determines optimal number of topics.
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
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

    def cluster_embeddings_bertopic(self, min_cluster_size: int = 30) -> Tuple[Dict, List[int]]:
        """Cluster emails using BERTopic instead of K-means.

        Args:
            min_cluster_size: Minimum emails per topic (default: 30)

        Returns:
            topic_info: BERTopic topic information
            topic_labels: Topic assignment per email
        """
        print("\n" + "="*60)
        print(f"BERTopic Clustering (auto-detect optimal topics)")
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

        hdbscan_model = HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )

        vectorizer_model = CountVectorizer(
            stop_words=list(EMAIL_STOPWORDS),
            ngram_range=(1, 3),  # Unigrams, bigrams, trigrams
            min_df=5  # Ignore rare words
        )

        # Create BERTopic model
        self.topic_model = BERTopic(
            embedding_model=None,  # Use pre-computed embeddings!
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            calculate_probabilities=True,
            verbose=True
        )

        # Fit model
        topics, probs = self.topic_model.fit_transform(documents, embeddings)

        # Get topic info
        topic_info = self.topic_model.get_topic_info()

        print(f"✓ Discovered {len(set(topics)) - 1} topics (excluding outliers)")
        print(f"  Outliers: {sum(1 for t in topics if t == -1)} emails")

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
                    output_path: Path = None, interactive: bool = False):
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
            min_cluster_size=min_cluster_size
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


def main():
    import argparse

    parser = argparse.ArgumentParser(description="BERTopic-based category discovery")
    parser.add_argument("--project", type=str,
                       default="data/projects/Primo_List")
    parser.add_argument("--min-cluster-size", type=int, default=30,
                       help="Minimum emails per topic")
    parser.add_argument("--output", type=str,
                       default="data/categories/discovered_categories_bertopic.json")
    parser.add_argument("--auto", action="store_true",
                       help="Non-interactive mode")

    args = parser.parse_args()

    print("="*60)
    print("  BERTOPIC EMAIL CATEGORY DISCOVERY")
    print("="*60)
    print(f"\nProject: {args.project}")
    print(f"Min cluster size: {args.min_cluster_size}")
    print(f"Output: {args.output}")

    project = ProjectManager(Path(args.project))
    discovery = BERTopicCategoryDiscovery(project)
    discovery.run_bertopic(
        min_cluster_size=args.min_cluster_size,
        output_path=Path(args.output),
        interactive=not args.auto
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

**Success Criteria:**
- ✅ Script runs without errors
- ✅ Auto-detects optimal number of topics
- ✅ Marks outliers as topic -1
- ✅ Generates meaningful topic keywords

---

### Task 2.3: Run BERTopic Discovery & Compare ⏱️ 30 min

**Actions:**

```bash
# 1. Run BERTopic discovery
PYTHONPATH=. poetry run python scripts/categorization/bertopic_discovery.py \
    --project data/projects/Primo_List \
    --min-cluster-size 30 \
    --output data/categories/discovered_categories_bertopic.json \
    --auto

# 2. Update evaluation script to use BERTopic results
# (modify evaluate_categories.py to load bertopic results)

# 3. Compare metrics
PYTHONPATH=. poetry run python scripts/categorization/evaluate_categories.py
```

**Comparison:**
| Metric | K-means (Phase 1) | BERTopic (Phase 2) | Improvement |
|--------|-------------------|-------------------|-------------|
| Silhouette | ~0.15 | ~0.35 | +133% |
| Balance | ~0.70 | ~0.80 | +14% |
| Coherence | ~0.40 | ~0.60 | +50% |
| Overall | ~0.42 | ~0.58 | +38% |

**Decision Point:**
- [ ] If BERTopic metrics better → proceed with BERTopic
- [ ] If K-means metrics better → investigate why, tune BERTopic params
- [ ] If similar → use BERTopic for outlier detection advantage

---

### Task 2.4: Update Discovery Pipeline to Use BERTopic ⏱️ 1 hour

**File:** `scripts/categorization/category_discovery.py`

**Action:** Make BERTopic the default, keep K-means as fallback

```python
# ADD command-line argument
parser.add_argument("--algorithm", type=str, default="bertopic",
                   choices=["kmeans", "bertopic"],
                   help="Clustering algorithm to use")

# MODIFY main()
if args.algorithm == "bertopic":
    from scripts.categorization.bertopic_discovery import BERTopicCategoryDiscovery
    discovery = BERTopicCategoryDiscovery(project)
    discovery.run_bertopic(...)
else:
    discovery = CategoryDiscovery(project)
    discovery.run(...)
```

---

### Phase 2 Validation Checklist

**Metrics:**
- [ ] Silhouette Score: ~0.35 (target: >0.25)
- [ ] Balance Score: ~0.80 (target: >0.75)
- [ ] Topic Coherence: ~0.60 (target: >0.60)
- [ ] Overall Quality: ~0.58 (target: >0.50)

**Quality Checks:**
- [ ] Sample 20 random emails per category → >80% correctly categorized
- [ ] No category has <30 emails (except outliers)
- [ ] Keywords are distinctive and meaningful
- [ ] Outliers cluster is <10% of total emails

**If Phase 2 successful:** Proceed to Phase 3

---

## 📋 PHASE 3: ADVANCED FEATURES (3-4 Days) → Target Quality: 0.68

**Goal:** Add thread-awareness, temporal analysis, hierarchical categories (+17% improvement)

**Expected Results:**
- Silhouette: 0.35 → 0.45 (+29%)
- Balance: 0.80 → 0.85 (+6%)
- Coherence: 0.60 → 0.75 (+25%)
- **Overall: 0.58 → 0.68 (+17%)**

---

### Task 3.1: Thread-Based Grouping ⏱️ 1 day

**Goal:** Group related emails into conversation threads, categorize threads instead of individual messages

**New File:** `scripts/categorization/thread_analyzer.py`

**Implementation:**

```python
import re
from collections import defaultdict
from typing import List, Dict
from scripts.chunking.models import Chunk


class EmailThreadAnalyzer:
    """Analyzes email threads for better categorization."""

    @staticmethod
    def normalize_subject(subject: str) -> str:
        """Normalize subject for thread matching.

        Removes:
        - [Primo], [ALMA-L] tags
        - Re:, Fwd: prefixes
        - Extra whitespace
        """
        # Remove mailing list tags
        normalized = re.sub(r'\[.*?\]', '', subject)
        # Remove re:, fwd: etc
        normalized = re.sub(r'^(re|fwd|fw):\s*', '', normalized, flags=re.IGNORECASE)
        # Normalize whitespace
        normalized = ' '.join(normalized.split())
        return normalized.lower().strip()

    def group_by_thread(self, emails: List[Chunk]) -> Dict[str, List[Chunk]]:
        """Group emails into conversation threads."""
        threads = defaultdict(list)

        for email in emails:
            subject = email.meta.get('subject', '')
            thread_key = self.normalize_subject(subject)
            threads[thread_key].append(email)

        return dict(threads)

    def categorize_thread(self, thread: List[Chunk],
                         categorizer) -> Tuple[str, float]:
        """Categorize entire thread (aggregate signals from all emails)."""
        # Combine all emails in thread
        combined_text = " ".join([
            email.meta.get('subject', '') + " " + email.text
            for email in thread
        ])

        # Create synthetic "thread email" for categorization
        thread_email = thread[0]  # Use first email as template
        thread_email.text = combined_text

        # Categorize
        category, confidence = categorizer.categorize(thread_email)

        return category, confidence

    def propagate_category_to_thread(self, thread: List[Chunk],
                                    category: str, confidence: float):
        """Apply category to all emails in thread."""
        for email in thread:
            email.meta['category'] = category
            email.meta['category_confidence'] = confidence
            email.meta['category_source'] = 'thread_analysis'
```

**Integration:**

```python
# IN bertopic_discovery.py, AFTER categorization

# Group into threads
thread_analyzer = EmailThreadAnalyzer()
threads = thread_analyzer.group_by_thread(self.email_chunks)

print(f"\n✓ Found {len(threads)} conversation threads")
print(f"  Avg thread length: {np.mean([len(t) for t in threads.values()]):.1f} emails")

# Re-categorize long threads for consistency
for thread_key, thread_emails in threads.items():
    if len(thread_emails) >= 3:  # Only for multi-email threads
        # Check if thread has mixed categories
        categories = [e.meta.get('category') for e in thread_emails]
        if len(set(categories)) > 1:  # Mixed categories
            # Re-categorize thread as whole
            category, conf = thread_analyzer.categorize_thread(
                thread_emails, self
            )
            thread_analyzer.propagate_category_to_thread(
                thread_emails, category, conf
            )
```

**Success Criteria:**
- ✅ Thread detection rate >70% (most emails grouped)
- ✅ Category consistency within threads increases
- ✅ Overall coherence improves

---

### Task 3.2: Temporal Topic Detection ⏱️ 1 day

**Goal:** Detect time-bound topics (e.g., "June 2025 Release Issues")

**New File:** `scripts/categorization/temporal_analyzer.py`

```python
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict
from collections import Counter


class TemporalTopicAnalyzer:
    """Detects temporal patterns in email topics."""

    def analyze_temporal_evolution(self, emails: List[Chunk],
                                   categories: Dict) -> Dict:
        """Analyze how topics evolve over time."""
        df = pd.DataFrame([
            {
                'date': pd.to_datetime(e.meta.get('date', '')),
                'category': e.meta.get('category', 'Unknown'),
                'text': e.text[:200]
            }
            for e in emails
        ])

        # Group by month and category
        df['month'] = df['date'].dt.to_period('M')

        temporal_topics = {}

        for category in df['category'].unique():
            cat_df = df[df['category'] == category]

            # Monthly keyword trends
            monthly_keywords = {}
            for month in cat_df['month'].unique():
                month_emails = cat_df[cat_df['month'] == month]
                # Extract top keywords for this month
                keywords = self.extract_monthly_keywords(month_emails['text'].tolist())
                monthly_keywords[str(month)] = keywords

            # Detect significant shifts
            if self.has_temporal_shift(monthly_keywords):
                temporal_topics[category] = {
                    'monthly_keywords': monthly_keywords,
                    'trend': 'evolving'
                }

        return temporal_topics

    def extract_monthly_keywords(self, texts: List[str]) -> List[str]:
        """Extract top keywords for a specific month."""
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(max_features=5, ngram_range=(1, 2))
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
            return vectorizer.get_feature_names_out().tolist()
        except:
            return []

    def has_temporal_shift(self, monthly_keywords: Dict) -> bool:
        """Detect if topic keywords change significantly over time."""
        if len(monthly_keywords) < 2:
            return False

        # Compare keyword overlap between consecutive months
        months = sorted(monthly_keywords.keys())
        overlaps = []

        for i in range(len(months) - 1):
            keywords1 = set(monthly_keywords[months[i]])
            keywords2 = set(monthly_keywords[months[i+1]])

            if keywords1 and keywords2:
                overlap = len(keywords1 & keywords2) / len(keywords1 | keywords2)
                overlaps.append(overlap)

        # If average overlap < 0.5, topic is shifting
        return np.mean(overlaps) < 0.5 if overlaps else False
```

**Success Criteria:**
- ✅ Detect at least 2-3 temporal topics
- ✅ Enable time-based queries ("What was discussed in June?")

---

### Task 3.3: Hierarchical Categories ⏱️ 1 day

**Goal:** Two-level category hierarchy (broad → specific)

**Implementation:**

```python
# IN bertopic_discovery.py

def create_hierarchical_categories(self, max_topics: int = 15):
    """Create 2-level hierarchy: discover many topics, then group."""

    # Step 1: Discover fine-grained topics (15-20)
    _, topics = self.cluster_embeddings_bertopic(min_cluster_size=20)

    # Step 2: Cluster topic centroids to create broad categories
    from sklearn.cluster import AgglomerativeClustering

    topic_centroids = np.array([
        self.category_centroids[tid]
        for tid in sorted(self.category_centroids.keys())
    ])

    # Hierarchical clustering on topics
    hierarchical = AgglomerativeClustering(n_clusters=5, linkage='ward')
    broad_categories = hierarchical.fit_predict(topic_centroids)

    # Build hierarchy
    hierarchy = {}
    for topic_id, broad_cat in zip(sorted(self.category_centroids.keys()),
                                   broad_categories):
        if broad_cat not in hierarchy:
            hierarchy[broad_cat] = []
        hierarchy[broad_cat].append(topic_id)

    # Name hierarchies
    for broad_id, specific_ids in hierarchy.items():
        # Get all emails in this broad category
        broad_emails = [e for e in self.email_chunks
                       if e.meta.get('cluster_id') in specific_ids]

        # Name broad category
        # ... (similar to LLM naming)

    return hierarchy
```

**Success Criteria:**
- ✅ 5 broad categories, each with 2-4 specific subcategories
- ✅ Enables multi-level queries

---

### Phase 3 Validation Checklist

**Metrics:**
- [ ] Silhouette Score: ~0.45 (target: >0.40)
- [ ] Balance Score: ~0.85 (target: >0.80)
- [ ] Topic Coherence: ~0.75 (target: >0.70)
- [ ] Overall Quality: ~0.68 (target: >0.60) ✅ **Production-ready**

**Feature Validation:**
- [ ] Thread detection: >70% emails grouped correctly
- [ ] Temporal topics: 2-3 detected successfully
- [ ] Hierarchical queries: Work correctly

**Human Evaluation:**
- [ ] Sample 50 emails: >85% correctly categorized
- [ ] Inter-annotator agreement: Cohen's Kappa >0.70
- [ ] User acceptance testing: Positive feedback on aggregation queries

**If Phase 3 successful:** System is production-ready!

---

## 🔧 Technical Components

### New Files to Create:

1. **`scripts/categorization/email_categorizer.py`**
   - EmailCategorizer class
   - 3-tier categorization logic
   - Confidence-based selection

2. **`scripts/categorization/category_discovery.py`**
   - Clustering logic
   - Pattern extraction
   - Centroid computation

3. **`scripts/retrieval/topic_aggregation_retriever.py`**
   - TopicAggregationRetriever class
   - Category-based grouping
   - Structured summary generation

4. **`scripts/categorization/category_config.json`**
   - Discovered categories
   - Rules
   - Centroids
   - Metadata

### Files to Modify:

1. **`scripts/ingestion/manager.py`**
   - Add categorization step
   - Store category in chunk metadata

2. **`scripts/agents/email_strategy_selector.py`**
   - Add topic_aggregation strategy
   - Map aggregation_query intent

3. **`scripts/agents/email_orchestrator.py`**
   - Add TopicAggregationRetriever
   - Handle aggregation_query intent

---

## 💰 Cost Analysis

### Query-Time vs Ingestion-Time (1000 emails, 10 aggregation queries)

**Query-Time Aggregation (Current Approach):**
- Ingestion: $0.10 (embeddings only)
- Each query: $0.05 (LLM analyzes 20 chunks)
- 10 queries: $0.60 total
- Speed: 30 seconds per query

**Ingestion-Time Categorization (Proposed):**
- Discovery: $0 (uses existing data)
- Categorization: $0.10 (10% use LLM)
- Embeddings: $0.10 (unchanged)
- Each query: $0.01 (one summary call)
- 10 queries: $0.30 total
- Speed: 2 seconds per query

**Savings: 50% cost, 15x faster!**

---

## 🎓 Key Learnings from Discussion

1. **Don't guess categories** - Discover them from actual data
2. **Leverage existing embeddings** - No need to recompute
3. **Confidence-based, not waterfall** - All tiers contribute
4. **Bootstrap from historical data** - Perfect for 6 months of emails
5. **Pay once, use forever** - Categorize at ingestion, benefit on every query

---

## 🧪 Experiments & Findings (2025-11-22)

### Context: Discovery Script Implementation

**Date:** November 22, 2025
**Dataset:** Primo_List project (1167 emails, 6 months of data)
**Objective:** Discover natural email categories using clustering + LLM naming

### Experiment 1: Initial Discovery Attempt ❌

**Approach:** K-means clustering (7 clusters) + simple keyword-based naming

**Implementation:**
- Used K-means on embeddings to create 7 clusters
- Named categories using pattern matching on subject keywords
- Logic: If "bug" in keywords → "Bug Reports", etc.

**Results:**
```
All 7 clusters named "Discussion"
Category collision: Last cluster overwrote all others
Final output: Only 1 category with 45 emails (should be 1167)
```

**Root Cause Analysis:**
1. **Keyword matching failed** - All Primo emails share "[primo]" prefix
2. **Insufficient differentiation** - Simple patterns couldn't distinguish clusters
3. **No duplicate detection** - Same name overwrote previous categories
4. **Clustering worked fine** - K-means found meaningful patterns:
   - Cluster 0: Announcements (webinar, registration, igelu)
   - Cluster 1: Technical questions (hover, display, import)
   - Cluster 2: Search issues (search, facet, availability)
   - Cluster 3: Release management (issue, release, records)
   - Cluster 4: Performance problems (performance, citation)
   - Cluster 5: Research Assistant (research, assistant, guardrails)
   - Cluster 6: Miscellaneous (small cluster)

**Conclusion:** Clustering algorithm is NOT the problem. Naming logic is the bottleneck.

---

### Solutions Considered

**Option 1: LLM-Based Cluster Naming** ⭐ SELECTED
- **Pros:** Intelligent analysis, cheap (~$0.01 for 7 clusters), works with K-means
- **Cons:** Adds API dependency, slight latency
- **Decision:** Best ROI - leverage LLM intelligence for minimal cost

**Option 2: HDBSCAN (Better Clustering)**
- **Pros:** Finds varying cluster sizes, auto-determines k, handles outliers
- **Cons:** Slower, doesn't solve naming problem, might find too many clusters
- **Decision:** Deferred - K-means working well enough

**Option 3: BERTopic (Topic Modeling)**
- **Pros:** Built for text, gives interpretable topics, works great with embeddings
- **Cons:** More complex, might be overkill, still needs naming
- **Decision:** Deferred - consider for v2 if K-means insufficient

**Option 4: Hybrid Approach (K-means + LLM + Duplicate Detection)**
- **Pros:** Combines speed of K-means with intelligence of LLM
- **Cons:** Requires LLM API access
- **Decision:** ✅ IMPLEMENTED

---

### Experiment 2: LLM-Based Naming ✅ SUCCESS

**Implementation Changes:**

1. **Added LLM Naming Method** (`_llm_name_cluster`)
   ```python
   def _llm_name_cluster(cluster_analysis, used_names):
       prompt = f"""Analyze this cluster and suggest category name:

       Size: {size} emails
       Keywords: {top_keywords}
       Sample subjects: {sample_subjects}

       Avoid these names: {used_names}
       Return 1-3 word category name."""

       return gpt35_turbo(prompt)
   ```

2. **Enhanced `name_categories()` Method**
   - Added `use_llm` parameter (default: True)
   - Track `used_names` list to avoid duplicates
   - If LLM returns duplicate, append cluster_id

3. **Added `--auto` CLI Flag**
   - Non-interactive mode for background execution
   - Auto-applies LLM suggestions without user prompts

4. **Fixed Path Issues**
   - Changed `project_dir` → `root_dir` (ProjectManager attribute)
   - Changed `embeddings/` → `faiss/` (actual directory name)

**Results:**

| Cluster | Size | LLM-Named Category | Keywords |
|---------|------|-------------------|----------|
| 0 | 184 | **Library Events** | 2025, call, enhancements, webinar, registration, igelu |
| 1 | 198 | **Discovery Systems** | hover, display, import, resource, discovery, profile |
| 2 | 301 | **Primo Support** | search, display, resource, facet, links, availability |
| 3 | 189 | **Primo Support (3)** | search, release, issue, records, results (Stacey's responses) |
| 4 | 119 | **Technical Support** | issues, release, citation, performance, chicago |
| 5 | 131 | **Research Assistant Support** | research, assistant, guardrails, filtering, recommender |
| 6 | 45 | **Email Management** | email reactions, miscellaneous |

**Observations:**
- ✅ All 7 categories received unique, meaningful names
- ✅ Duplicate detected (Cluster 3 = "Primo Support (3)")
- ✅ LLM understood context from keywords + sample subjects
- ✅ Total cost: ~$0.007 (7 clusters × ~$0.001/call)
- ✅ Rules extracted (keywords appearing in >30% of emails)
- ✅ Centroids computed for Tier 2 categorization

**Sample Category Analysis:**

**Library Events (184 emails):**
- Top keywords: [primo], 2025, [alma-l], call, enhancements, webinar
- Top senders: Stacey van Groll, Tamar Ganor, Nili Natan
- Date range: 2025-05-26 to 2025-11-19
- **Interpretation:** Announcements, conferences, enhancement ballots

**Research Assistant Support (131 emails):**
- Top keywords: research, assistant (53.4% of subjects!), guardrails, filtering
- Common body words: research, search, assistant, primo, about
- **Interpretation:** Dedicated cluster for Primo Research Assistant feature

**Technical Support (119 emails):**
- Top keywords: issues, release, july, citation, performance, chicago
- Senders: Sima Bloch-Winkler (11), Amy Pemble (8)
- **Interpretation:** Performance issues, bugs, citation problems

---

### Technical Implementation Details

**Files Modified:**
```
scripts/categorization/category_discovery.py (+119 lines)
  - Added OpenAICompleter import
  - Added _llm_name_cluster() method (57 lines)
  - Enhanced name_categories() with LLM support
  - Added duplicate detection logic
  - Added --auto flag for non-interactive mode
  - Fixed attribute name bugs (project_dir → root_dir)
  - Fixed path bug (embeddings → faiss)
```

**Files Created:**
```
data/categories/discovered_categories.json (87 KB)
  - 7 unique categories with mapping
  - Rules for each category (subject + body keywords)
  - Centroids for each category (1536-dim embeddings)
  - Counts and confidence scores
  - Discovery metadata (date, project, totals)
```

**Git:**
```
Branch: feature/email-categorization
Commit: 54bed15 - feat(categorization): Implement LLM-based cluster naming
Status: Clean, ready for next phase
```

---

### Performance Metrics

**Discovery Script Execution:**
- Load 1167 chunks: ~1 second
- Load FAISS embeddings: ~2 seconds
- K-means clustering (n=7): ~3 seconds
- Cluster analysis (7 clusters): ~2 seconds
- LLM naming (7 calls): ~10 seconds
- Rule extraction: ~1 second
- Centroid computation: ~1 second
- **Total time:** ~20 seconds

**Cost Analysis:**
- K-means clustering: $0 (local computation)
- LLM naming: ~$0.007 (7 × $0.001)
- Embedding reuse: $0 (already computed)
- **Total cost:** $0.007

**Quality Assessment:**
- Unique categories: 7/7 (100%)
- Meaningful names: 7/7 (subjective, but clear)
- Rule coverage: All categories have 1-3 subject keywords
- Centroid quality: Based on 45-301 emails per cluster

---

### Comparison: Before vs After

| Aspect | Before (Keyword Matching) | After (LLM Naming) |
|--------|---------------------------|-------------------|
| **Unique categories** | 1 | 7 |
| **Name quality** | Generic ("Discussion") | Specific ("Library Events", "Technical Support") |
| **Duplicate handling** | ❌ Overwrites | ✅ Appends cluster_id |
| **Context awareness** | Keywords only | Keywords + sample subjects + body patterns |
| **Cost** | $0 | $0.007 |
| **Speed** | <1s | ~10s (LLM calls) |
| **Usability** | ❌ Failed | ✅ Production-ready |

---

### Lessons Learned

1. **K-means is sufficient** - No need for HDBSCAN/BERTopic yet
   - Found natural groupings in Primo mailing list data
   - 7 clusters balanced sizes (45-301 emails)
   - Clear topical separation

2. **LLM naming is game-changer** - Small cost, huge quality gain
   - $0.007 to name 7 clusters is negligible
   - Context-aware naming vastly superior to keyword matching
   - Handles domain-specific patterns (Primo, IGeLU, Research Assistant)

3. **Duplicate detection is critical** - Must track used names
   - 2 clusters both matched "Primo Support" pattern
   - Automatic suffix prevents collision
   - Could improve with better prompting or re-try logic

4. **Embeddings reuse = free clustering** - No re-computation needed
   - Leveraged existing FAISS index
   - 1167 embeddings already available
   - Discovery cost = $0 (only naming costs)

5. **Data reveals patterns humans miss** - Inductive > Deductive
   - Didn't expect "Research Assistant" to be its own cluster
   - "Library Events" cluster (announcements) emerged naturally
   - "Email Management" cluster caught email system messages

---

### Known Issues & Future Improvements

**Issue 1: Cluster 3 Duplicate Name**
- LLM suggested "Primo Support" again (same as Cluster 2)
- Auto-appended "(3)" to make unique
- **Better approach:** Re-prompt LLM with more context or stronger constraint

**Issue 2: Generic Category Names**
- "Primo Support" and "Discovery Systems" are somewhat vague
- **Better approach:** Second LLM pass to refine names, or human review step

**Issue 3: Cluster Size Imbalance**
- Cluster 2 (301 emails) vs Cluster 6 (45 emails)
- **Investigation needed:** Is 45-email cluster meaningful or noise?
- **Potential fix:** HDBSCAN could identify this as outliers

**Issue 4: Category Evolution Not Implemented**
- Current: Fixed 7 categories
- Needed: Detect when new categories emerge
- **Future work:** Monitor "Other" category, suggest splits

**Issue 5: No Validation Yet**
- Haven't manually validated categorization accuracy
- **Next step:** Sample 50 emails, human labeling, measure accuracy

---

### Recommended Next Steps

**Immediate (Phase 2):**
1. ✅ Update this documentation (DONE)
2. Implement `EmailCategorizer` class (3-tier logic)
3. Test categorization on sample emails
4. Integrate into `IngestionManager`

**Short-term (Phase 3):**
5. Create `TopicAggregationRetriever`
6. Update `EmailStrategySelector` for aggregation queries
7. Test end-to-end aggregation flow

**Validation (Phase 4):**
8. Manual validation: Sample 50 emails, measure accuracy
9. A/B test: Compare with/without categorization
10. Tune confidence thresholds based on validation

**Future Enhancements:**
- Try HDBSCAN to see if it finds better clusters
- Try BERTopic for interpretable topics
- Implement category evolution (detect new topics)
- Add UI filters by category
- Add analytics dashboard (category trends over time)

---

## 📚 References

- Email Phase 4 Completion: `docs/archive/EMAIL_PHASE4_COMPLETION.md`
- Email Agentic Strategy: `docs/automation/EMAIL_AGENTIC_STRATEGY_MERGED.md`
- Master Roadmap: `docs/MASTER_ROADMAP.md`

---

**Status:** Baseline Established (Quality: 0.254) → Phase 1 Ready
**Branch:** `feature/email-categorization`
**Next Step:** Execute Phase 1 Quick Wins

---

## ✅ IMMEDIATE NEXT ACTIONS

### Priority 1: Execute Phase 1 (START NOW)

**Estimated Time:** 2-3 hours
**Expected Improvement:** 0.254 → 0.42 (+65%)

**Tasks in Order:**

1. **Task 1.3: Filter System Artifacts** (30 min)
   - Add `_is_system_artifact()` method
   - Filter reaction emails and auto-replies
   - Most impactful, do first

2. **Task 1.2: Implement TF-IDF Keywords** (1 hour)
   - Replace frequency counting with TF-IDF
   - Extract distinctive keywords per category
   - Include bigrams ("research assistant")

3. **Task 1.1: Merge Categories** (30 min)
   - Re-run with `--n-categories 5`
   - Consolidate overlapping categories

4. **Validation** (30 min)
   - Run evaluation script
   - Check metrics meet targets
   - Sample emails for qualitative check

**Commands to Run:**

```bash
# 1. Update code (Tasks 1.2 and 1.3)
# Edit scripts/categorization/category_discovery.py

# 2. Re-run discovery with improvements
PYTHONPATH=. poetry run python scripts/categorization/category_discovery.py \
    --project data/projects/Primo_List \
    --n-categories 5 \
    --output data/categories/discovered_categories_phase1.json \
    --auto

# 3. Evaluate
PYTHONPATH=. poetry run python scripts/categorization/evaluate_categories.py
```

**Success Criteria:**
- [ ] Overall Quality: ~0.42 (currently 0.254)
- [ ] Silhouette: ~0.15 (currently 0.026)
- [ ] "Email Management" category removed/reduced
- [ ] Keywords more meaningful (phrases, not just single words)

---

### Priority 2: Plan Phase 2 (BERTopic)

**After Phase 1 succeeds:**

1. Install dependencies: `poetry add bertopic umap-learn hdbscan`
2. Create `bertopic_discovery.py` (use template from plan)
3. Test on Primo_List dataset
4. Compare metrics with Phase 1 baseline

**Target:** Quality 0.42 → 0.58 (+38%)

---

### Priority 3: Integrate into Production

**After Phase 2/3 complete (Quality >0.60):**

1. Implement `EmailCategorizer` class (3-tier logic)
2. Integrate into `IngestionManager`
3. Add category metadata to chunks during ingestion
4. Test on new incoming emails
5. Deploy to production

---

## 📊 PROGRESS TRACKING

### Baseline Metrics (Established 2025-11-22)

| Metric | Value | Status |
|--------|-------|--------|
| Silhouette Score | 0.026 | ⚠️ Poor |
| Balance Score | 0.557 | ⚠️ Fair |
| Topic Coherence | 0.256 | ⚠️ Fair |
| **Overall Quality** | **0.254** | **⚠️ Poor** |

### Phase 1 Targets

| Metric | Baseline | Target | Status |
|--------|----------|--------|--------|
| Silhouette | 0.026 | 0.15 | ⏳ Pending |
| Balance | 0.557 | 0.70 | ⏳ Pending |
| Coherence | 0.256 | 0.40 | ⏳ Pending |
| **Overall** | **0.254** | **0.42** | **⏳ Pending** |

### Phase 2 Targets (BERTopic)

| Metric | Phase 1 | Target | Status |
|--------|---------|--------|--------|
| Silhouette | 0.15 | 0.35 | 🔒 Locked |
| Balance | 0.70 | 0.80 | 🔒 Locked |
| Coherence | 0.40 | 0.60 | 🔒 Locked |
| **Overall** | **0.42** | **0.58** | **🔒 Locked** |

### Phase 3 Targets (Advanced)

| Metric | Phase 2 | Target | Status |
|--------|---------|--------|--------|
| Silhouette | 0.35 | 0.45 | 🔒 Locked |
| Balance | 0.80 | 0.85 | 🔒 Locked |
| Coherence | 0.60 | 0.75 | 🔒 Locked |
| **Overall** | **0.58** | **0.68** | **🔒 Locked** |

---

## 🎓 KEY LEARNINGS FROM EVALUATION

### What Worked ✅

1. **LLM-based cluster naming** - Generated meaningful category names
2. **Comprehensive keyword filtering** - 480 person names blocked successfully
3. **Evaluation framework** - Automated metrics enable rapid iteration
4. **Research-driven approach** - BERTopic recommendation backed by literature

### What Didn't Work ❌

1. **K-means clustering** - Wrong algorithm for mailing lists (silhouette: 0.026)
2. **Simple frequency counting** - Missed distinctive keywords
3. **7 categories** - Too many overlapping categories
4. **30% threshold** - Too strict after aggressive filtering

### Critical Insights 💡

1. **Mailing lists are challenging** - Diverse topics, overlapping discussions
2. **Context matters** - Email threads should be analyzed together
3. **Temporal patterns exist** - Some topics are time-bound (releases, events)
4. **Quality > Quantity** - 5 well-defined categories better than 7 fuzzy ones

---

## 🔗 RELATED DOCUMENTATION

- **Evaluation Report:** `docs/future/EMAIL_CATEGORIZATION_EVALUATION.md`
  - Full analysis of current performance
  - Research findings (BERTopic, TF-IDF, metrics)
  - Detailed improvement strategies

- **Test Results:** `data/categories/discovered_categories.json`
  - 7 categories discovered from 1167 emails
  - Baseline for comparison

- **Evaluation Script:** `scripts/categorization/evaluate_categories.py`
  - Automated metrics calculation
  - Silhouette, Balance, Coherence scores
  - Sample email analysis

- **Discovery Script:** `scripts/categorization/category_discovery.py`
  - K-means clustering + LLM naming
  - Keyword filtering (stopwords + person names)
  - Current baseline implementation

---

## ✅ READY TO PROCEED

**Current State:** All groundwork complete, ready for Phase 1 implementation

**Recommended Action:** Start Phase 1 now (2-3 hours) → Expected +65% quality improvement

**After Phase 1:** Evaluate results, then proceed to Phase 2 (BERTopic) if successful

**Final Goal:** Production-ready categorization system (Quality >0.60) within 1-2 weeks

---

**Last Updated:** 2025-11-22
**Status:** Operational Plan Ready for Execution
**Owner:** Email Categorization Feature Team
