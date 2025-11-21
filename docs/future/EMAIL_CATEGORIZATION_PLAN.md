# Email Categorization System Plan

**Date:** 2025-11-21
**Priority:** HIGH
**Status:** Planning → Implementation
**Estimated Effort:** 1 week implementation, continuous refinement

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

## 🚀 Implementation Plan

### Week 1: Discovery & Bootstrap
- [x] Document plan (this document)
- [ ] Implement category discovery script
- [ ] Run discovery on 6 months of data
- [ ] Analyze clusters and name categories
- [ ] Extract rules and compute centroids
- [ ] Save discovered categories

### Week 2: Categorization System
- [ ] Implement 3-tier categorization
- [ ] Integrate into ingestion pipeline
- [ ] Add category metadata to chunks
- [ ] Test on sample emails

### Week 3: Query Integration
- [ ] Create TopicAggregationRetriever
- [ ] Update EmailStrategySelector
- [ ] Integrate with EmailOrchestratorAgent
- [ ] Test aggregation queries

### Week 4: Validation & Refinement
- [ ] Re-categorize existing 6 months (optional)
- [ ] Validate accuracy (sampling)
- [ ] Tune confidence thresholds
- [ ] Document final system

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

## 📚 References

- Email Phase 4 Completion: `docs/archive/EMAIL_PHASE4_COMPLETION.md`
- Email Agentic Strategy: `docs/automation/EMAIL_AGENTIC_STRATEGY_MERGED.md`
- Master Roadmap: `docs/MASTER_ROADMAP.md`

---

**Status:** Ready for implementation
**Next Step:** Run category discovery script on Primo_List project
