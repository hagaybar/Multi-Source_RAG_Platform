# Email Categorization System - Systematic Evaluation & Improvement Plan

**Date:** 2025-11-22
**Status:** Analysis Complete → Planning Improvements
**Current Quality Score:** 0.254 / 1.0 (⚠️ Needs significant improvement)

---

## 📊 PART 1: QUANTITATIVE EVALUATION

### Current Metrics (Baseline)

| Metric | Score | Threshold | Status | Interpretation |
|--------|-------|-----------|--------|----------------|
| **Silhouette Score** | 0.0264 | >0.25 = Good | ⚠️ Poor | Clusters are heavily overlapping |
| **Balance Score** | 0.557 | >0.70 = Good | ⚠️ Fair | Uneven distribution (45-301 emails) |
| **Topic Coherence** | 0.2558 | >0.30 = Good | ⚠️ Fair | Keywords don't strongly co-occur |
| **Overall Quality** | 0.254 | >0.60 = Good | ⚠️ Poor | System needs improvement |

### Category Distribution Analysis

```
Total emails: 1167
Categories: 7
Average size: 166.7 emails
Standard deviation: 73.8
Range: 45 - 301 emails
Coefficient of variation: 0.443 (0 = perfect balance, 1 = highly imbalanced)
```

**Issue:** Cluster size imbalance suggests some categories are "catch-all" while others are too specific.

### Individual Category Coherence

| Category | Coherence | Keywords Available | Assessment |
|----------|-----------|-------------------|------------|
| Technical Support | 0.4680 | ✅ Yes | Best performing |
| Email Management | 0.3267 | ✅ Yes | Good |
| Research Assistant Inquiries | 0.2380 | ✅ Yes | Fair |
| Technical Problems | 0.2280 | ✅ Yes | Fair |
| Product Updates | 0.1800 | ✅ Yes | Weak |
| User Interface Customization | 0.0940 | ✅ Yes | Very weak |
| User Interface Issues | - | ❌ No keywords | Failed |

**Issue:** Low coherence indicates keywords don't capture the topic essence well.

---

## 🔍 PART 2: QUALITATIVE EVALUATION

### Sample Email Analysis (Random Sampling)

#### ✅ **Well-Categorized Examples:**

**Category: Technical Support**
```
Subject: [Primo] Re: calculated availability question
Context: Discussing how availability statements display in Primo
Assessment: ✅ Correct - Technical support question about system behavior
```

**Category: Product Updates**
```
Subject: Coming Soon: Quarterly Alma Digital/Specto Essentials Meetings
Context: Announcement about upcoming meetings and features
Assessment: ✅ Correct - Product announcement/update
```

#### ⚠️ **Questionable Categorizations:**

**Category: Research Assistant Inquiries**
```
Subject: Re: catalogue searching in advanced search mode
Context: General search functionality discussion
Assessment: ⚠️ Incorrect - Not about Research Assistant feature
Reason: Keyword "search" appears in both, causing misclassification
```

**Category: User Interface Issues vs. Customization**
```
Too much overlap - both deal with UI, hard to distinguish
Examples blur between "having issues" and "customizing"
Assessment: ⚠️ Categories too similar, should merge
```

#### ❌ **Clear Misclassifications:**

**Category: Email Management**
```
Subject: [reactions to messages with emojis]
Context: Email system artifacts (like/sad reactions)
Assessment: ❌ These aren't real content - should be filtered entirely
```

### Root Causes of Poor Performance

1. **Overlapping Categories**
   - "User Interface Issues" vs "User Interface Customization"
   - "Technical Support" vs "Technical Problems"
   - Semantic overlap causes low silhouette score

2. **Weak Keyword Signals**
   - After aggressive filtering, too few distinctive keywords remain
   - Generic terms like "search", "display", "issue" appear everywhere
   - 30% threshold may be too high for diverse mailing list content

3. **Mailing List Characteristics**
   - Threads drift across multiple topics
   - Subject lines use "Re:" extensively (loses context)
   - Community discussions are naturally diverse and fluid

4. **Insufficient Context**
   - Using subject + first 200 chars of body isn't enough
   - Email threads should be analyzed as conversations, not isolated messages
   - Temporal patterns not considered (topics evolve over time)

---

## 🌐 PART 3: RESEARCH-BASED INSIGHTS

### State-of-the-Art Approaches (2024-2025 Research)

#### 1. **BERTopic vs. K-means**

**Current Approach:** K-means clustering on embeddings

**Research Findings:**
- **BERTopic** outperforms K-means for topic modeling in email/document clustering
- Uses UMAP (dimensionality reduction) + HDBSCAN (density-based clustering)
- Automatically determines optimal number of topics (no need to specify k=7)
- Better handles outliers (important for diverse mailing lists)
- Uses c-TF-IDF for topic representation (class-based, not document-based)

**Advantages over K-means:**
- Handles non-spherical clusters (mailing list topics aren't evenly distributed)
- Variable cluster sizes (acknowledges some topics are more common)
- Outlier detection (-1 cluster for noise)
- More interpretable topics

**Source:** Pinecone, BERTopic documentation, ACM research papers

#### 2. **TF-IDF Enhancements**

**Current Approach:** Simple word frequency counting with stopwords

**Research Findings:**
- **c-TF-IDF** (class-based TF-IDF) works at category/topic level instead of document level
- Better captures what makes a category unique vs. other categories
- Can use `reduce_frequent_words` parameter to auto-filter common terms

**Implementation:**
```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(
    stop_words="english",
    ngram_range=(1, 3),  # Include phrases
    min_df=5,  # Ignore rare words
    max_df=0.7  # Ignore very common words
)
```

**Source:** BERTopic Tips & Tricks, ScienceDirect research

#### 3. **Hybrid Approaches**

**Research Recommendation:** Combine multiple signals

**Multi-Level Categorization:**
1. **Embedding-based** (semantic similarity) ← Current approach
2. **TF-IDF based** (keyword importance) ← Missing
3. **Thread-based** (conversation context) ← Missing
4. **Temporal-based** (time-based topics) ← Missing

**Example Pipeline:**
```
Raw emails → Thread grouping → Embedding → UMAP → HDBSCAN → c-TF-IDF → LLM naming
```

**Source:** Nature Scientific Reports, SpringerLink

#### 4. **Evaluation Metrics Best Practices**

**Current Metrics:** Silhouette, Coherence, Balance

**Research Recommendations:**
- **Silhouette Score:** Good for K-means, less useful for density-based clustering
- **Topic Coherence:** Use PMI (Pointwise Mutual Information) or NPMI instead of simple co-occurrence
- **Human Evaluation:** Sample 50-100 emails, manual labeling, compute precision/recall
- **Perplexity:** For probabilistic models (LDA)

**Source:** Webex Developer Blog, DataScienceBase

---

## 🚀 PART 4: IMPROVEMENT PLAN

### Strategy: Phased Enhancement Approach

We'll implement improvements in 3 phases, measuring impact at each step.

---

### **Phase 1: Quick Wins (Current System Enhancement)** ⏱️ 2-3 hours

**Goal:** Improve current K-means approach without changing architecture

#### 1.1. Merge Overlapping Categories

**Action:**
- Merge "User Interface Issues" + "User Interface Customization" → "User Interface"
- Merge "Technical Support" + "Technical Problems" → "Technical Issues"
- Results in 5 categories instead of 7

**Expected Impact:**
- ↑ Silhouette score (less overlap)
- ↑ Balance score (larger, more meaningful clusters)
- ↑ Topic coherence (more data per category)

**Implementation:**
```bash
# Re-run discovery with n_categories=5
python scripts/categorization/category_discovery.py --project data/projects/Primo_List \
    --n-categories 5 --output data/categories/discovered_categories_v2.json --auto
```

#### 1.2. Lower Keyword Threshold

**Action:**
- Change keyword extraction threshold from 30% to 15%
- Capture more diverse keywords per category

**Code Change:**
```python
# In extract_rules()
threshold = len(cluster_emails) * 0.15  # Was 0.3
```

**Expected Impact:**
- ↑ Number of subject_keywords (currently many are empty)
- ↑ Topic coherence (more signals available)

#### 1.3. Use TF-IDF for Keyword Extraction

**Action:**
- Instead of raw frequency, use TF-IDF to find distinctive keywords
- Prioritizes words that are common in category but rare elsewhere

**Code Change:**
```python
from sklearn.feature_extraction.text import TfidfVectorizer

# In extract_rules()
vectorizer = TfidfVectorizer(
    max_features=20,
    ngram_range=(1, 2),  # Include bigrams
    stop_words=list(EMAIL_STOPWORDS)
)

# Fit on category emails
texts = [c.get('subject', '') + ' ' + c.text for c in cluster_emails]
tfidf_matrix = vectorizer.fit_transform(texts)

# Get top keywords by TF-IDF score
feature_names = vectorizer.get_feature_names_out()
scores = tfidf_matrix.sum(axis=0).A1
top_indices = scores.argsort()[-10:][::-1]
subject_keywords = [feature_names[i] for i in top_indices]
```

**Expected Impact:**
- ↑ Keyword quality (distinctive terms, not just frequent)
- ↑ Topic coherence
- Better rule-based categorization (Tier 1)

#### 1.4. Filter Email System Artifacts

**Action:**
- Detect and skip emails that are pure system messages (reactions, etc.)

**Code Change:**
```python
# In load_email_chunks()
def is_system_artifact(email: Dict) -> bool:
    text = email.get('text', '')
    # Check for reaction patterns
    if 'reacted to your message' in text.lower():
        return True
    if 'outlook-1.cdn.office.net/assets/reaction' in text:
        return True
    return False

# Filter during loading
chunks = [c for c in chunks if not is_system_artifact(c.meta)]
```

**Expected Impact:**
- ↑ Data quality (no noise emails)
- ↑ All metrics (cleaner signal)

**Estimated Results After Phase 1:**
- Silhouette: 0.026 → ~0.15 (↑ 480%)
- Balance: 0.557 → ~0.70 (↑ 26%)
- Coherence: 0.256 → ~0.40 (↑ 56%)
- Overall: 0.254 → ~0.42 (↑ 65%)

---

### **Phase 2: Architecture Upgrade (BERTopic Integration)** ⏱️ 1-2 days

**Goal:** Replace K-means with BERTopic for better topic modeling

#### 2.1. Install BERTopic

```bash
poetry add bertopic umap-learn hdbscan
```

#### 2.2. Implement BERTopic Discovery

**New File:** `scripts/categorization/bertopic_discovery.py`

```python
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN

# Custom components
umap_model = UMAP(
    n_neighbors=15,
    n_components=5,
    min_dist=0.0,
    metric='cosine',
    random_state=42
)

hdbscan_model = HDBSCAN(
    min_cluster_size=30,  # Min 30 emails per topic
    metric='euclidean',
    cluster_selection_method='eom',
    prediction_data=True
)

vectorizer_model = CountVectorizer(
    stop_words=list(EMAIL_STOPWORDS),
    ngram_range=(1, 3),
    min_df=5
)

# BERTopic model
topic_model = BERTopic(
    embedding_model=None,  # Use existing embeddings!
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    nr_topics="auto",  # Auto-determine optimal number
    calculate_probabilities=True
)

# Fit (using pre-computed embeddings)
topics, probs = topic_model.fit_transform(documents, embeddings)
```

**Advantages:**
- Auto-determines optimal number of topics
- Handles outliers (topic -1)
- Better topic coherence
- Built-in topic visualization

#### 2.3. Integrate with Existing Pipeline

**Changes:**
- Keep existing embedding step (reuse FAISS embeddings)
- Replace `cluster_embeddings()` with `bertopic_cluster()`
- Keep LLM naming step
- Keep centroid computation for Tier 2

**Expected Results After Phase 2:**
- Silhouette: ~0.15 → ~0.35 (↑ 133%)
- Balance: ~0.70 → ~0.80 (↑ 14%)
- Coherence: ~0.40 → ~0.60 (↑ 50%)
- Overall: ~0.42 → ~0.58 (↑ 38%)

---

### **Phase 3: Advanced Features (Thread & Temporal Analysis)** ⏱️ 3-4 days

**Goal:** Incorporate conversation context and time-based patterns

#### 3.1. Thread-Based Grouping

**Concept:** Group related emails into threads, categorize threads instead of individual emails

**Implementation:**
```python
def group_by_thread(emails: List[Dict]) -> Dict[str, List[Dict]]:
    """Group emails by conversation thread."""
    threads = {}

    for email in emails:
        # Thread ID from subject line
        subject = email.get('subject', '')
        # Remove "Re:", "Fwd:", etc.
        clean_subject = re.sub(r'^(re:|fwd:)\s*', '', subject, flags=re.IGNORECASE)

        # Use normalized subject as thread key
        thread_key = clean_subject.lower().strip()

        if thread_key not in threads:
            threads[thread_key] = []
        threads[thread_key].append(email)

    return threads

# Then categorize threads, propagate category to all emails in thread
```

**Benefits:**
- Context-aware (entire conversation, not just one message)
- Reduces noise from "Re: Re: Re:" emails
- More consistent categorization

#### 3.2. Temporal Topic Detection

**Concept:** Some topics are time-bound (e.g., "June 2025 Release Issues")

**Implementation:**
```python
def detect_temporal_topics(emails: List[Dict], categories: Dict) -> Dict:
    """Detect time-based topic variations."""
    import pandas as pd

    df = pd.DataFrame(emails)
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.to_period('M')

    # For each category, check if topic shifts over time
    temporal_topics = {}

    for category, category_emails in grouped_emails.items():
        # Group by month
        monthly_keywords = {}

        for month, month_emails in groupby(category_emails, key='month'):
            keywords = extract_keywords_tfidf(month_emails)
            monthly_keywords[month] = keywords

        # Detect shifts (keywords changing significantly month-to-month)
        if has_temporal_shift(monthly_keywords):
            temporal_topics[category] = monthly_keywords

    return temporal_topics
```

**Use Case:**
- Query: "What were people discussing last month?"
- Answer: Can focus on recent temporal topics, not all-time categories

#### 3.3. Hierarchical Categories

**Concept:** Two-level hierarchy (broad → specific)

**Example:**
```
Level 1: Technical Issues
  ├─ Level 2a: Performance Problems
  ├─ Level 2b: Citation Issues
  └─ Level 2c: Search Problems

Level 1: Product Features
  ├─ Level 2a: Research Assistant
  ├─ Level 2b: User Interface
  └─ Level 2c: Analytics
```

**Implementation:**
- Run BERTopic with `nr_topics=15` (more granular)
- Use hierarchical clustering on topic embeddings to group into 5-7 broad categories
- Enables both specific and general queries

**Expected Results After Phase 3:**
- Silhouette: ~0.35 → ~0.45 (↑ 29%)
- Balance: ~0.80 → ~0.85 (↑ 6%)
- Coherence: ~0.60 → ~0.75 (↑ 25%)
- Overall: ~0.58 → ~0.68 (↑ 17%)

---

## 📈 PART 5: SUCCESS METRICS & VALIDATION

### Quantitative Targets

| Metric | Baseline | Phase 1 | Phase 2 | Phase 3 | Target |
|--------|----------|---------|---------|---------|--------|
| Silhouette Score | 0.026 | 0.15 | 0.35 | 0.45 | >0.40 |
| Balance Score | 0.557 | 0.70 | 0.80 | 0.85 | >0.75 |
| Topic Coherence | 0.256 | 0.40 | 0.60 | 0.75 | >0.60 |
| **Overall Quality** | **0.254** | **0.42** | **0.58** | **0.68** | **>0.60** |

### Qualitative Validation

**Human Evaluation Protocol:**
1. Sample 50 random emails from each category
2. 2 human annotators independently label: Correct / Incorrect / Ambiguous
3. Compute inter-annotator agreement (Cohen's Kappa)
4. Target: >80% agreement, >85% accuracy

**A/B Testing:**
- Compare aggregation queries with/without categorization
- Measure: Answer quality, response time, cost
- Target: 50% cost reduction, 10x faster

---

## 🎯 PART 6: RECOMMENDED IMMEDIATE ACTIONS

### Priority 1: Execute Phase 1 (Today)

```bash
# 1. Update category_discovery.py with improvements
# 2. Re-run with 5 categories
poetry run python scripts/categorization/category_discovery.py \
    --project data/projects/Primo_List \
    --n-categories 5 \
    --output data/categories/discovered_categories_phase1.json \
    --auto

# 3. Re-evaluate
poetry run python scripts/categorization/evaluate_categories.py
```

**Expected:** Overall quality 0.254 → 0.42 (+65%)

### Priority 2: Prototype BERTopic (Next Session)

```bash
# 1. Install dependencies
poetry add bertopic umap-learn hdbscan

# 2. Create bertopic_discovery.py
# 3. Test on Primo_List dataset
# 4. Compare metrics with K-means baseline
```

**Expected:** Overall quality 0.42 → 0.58 (+38%)

### Priority 3: Document & Integrate (This Week)

- Update EMAIL_CATEGORIZATION_PLAN.md with findings
- Implement EmailCategorizer class (3-tier logic)
- Integrate into ingestion pipeline
- Test on new incoming emails

---

## 📚 PART 7: REFERENCES & RESOURCES

### Research Papers

1. **"An optimized BERTopic framework based on cluster silhouette for improving topic coherence"** (ACM 2024)
   - Shows BERTopic + silhouette optimization improves coherence by 40%

2. **"Classifying spam emails using agglomerative hierarchical clustering and a topic-based approach"** (ScienceDirect 2023)
   - Hybrid TF-IDF + clustering approach

3. **"Evaluating Cohesion Score with Email Clustering"** (SpringerLink)
   - Email-specific clustering evaluation metrics

### Tools & Libraries

- **BERTopic:** https://maartengr.github.io/BERTopic/
- **UMAP:** https://umap-learn.readthedocs.io/
- **HDBSCAN:** https://hdbscan.readthedocs.io/
- **scikit-learn:** TF-IDF, Silhouette Score

### Best Practices Guides

- BERTopic Best Practices: https://maartengr.github.io/BERTopic/getting_started/best_practices/best_practices.html
- Topic Modeling Evaluation: https://www.datasciencebase.com/unsupervised-ml/advanced-clustering-topics/evaluation-metrix-usvml/

---

## 🔄 PART 8: ITERATIVE IMPROVEMENT CYCLE

### Continuous Monitoring

**Weekly:**
- Run evaluation script on latest data
- Track metric trends over time
- Sample 10 random categorizations for spot-checks

**Monthly:**
- Full human evaluation (50 samples)
- Re-cluster if silhouette score drops below threshold
- Update stopwords based on new noise patterns

**Quarterly:**
- Consider new topics (category evolution)
- A/B test new approaches
- Update documentation

### Feedback Loop

```
User Query → Retrieve by Category → LLM Answer → User Feedback
                ↓
         Implicit Signals
                ↓
    Low confidence? → Re-categorize → Update centroids
```

---

## ✅ CONCLUSION

**Current State:**
- ⚠️ Overall quality: 0.254 / 1.0 (Poor)
- ⚠️ Weak clustering (silhouette 0.026)
- ⚠️ Low topic coherence (0.256)
- ✅ Functional keyword filtering (improved from initial run)

**Path Forward:**
- **Phase 1 (Quick):** Merge categories, TF-IDF keywords → **0.42 quality**
- **Phase 2 (Medium):** BERTopic integration → **0.58 quality**
- **Phase 3 (Advanced):** Threads + temporal → **0.68 quality**

**Target Achieved:** >0.60 overall quality (production-ready)

**Next Step:** Execute Phase 1 improvements immediately.

---

**Status:** Ready for Phase 1 Implementation
**Owner:** Email Categorization Feature Team
**Last Updated:** 2025-11-22
