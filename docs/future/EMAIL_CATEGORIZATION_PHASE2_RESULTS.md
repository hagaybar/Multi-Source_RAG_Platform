# Email Categorization - Phase 2 Results (BERTopic)

**Date:** 2025-11-23
**Project:** Primo_List_2 Email Dataset
**Scope:** Phase 2 BERTopic Integration & Evaluation
**Status:** ⚠️ COMPLETED - Unexpected Results Require Investigation

---

## Executive Summary

Phase 2 implemented BERTopic clustering as planned, but produced **significantly worse results** than Phase 1 K-means. The overall quality score **declined from 0.371 to 0.075** (-80%), contradicting research expectations. This requires investigation before proceeding.

### Key Findings

❌ **Challenges:**
- Overall quality: 0.371 → 0.075 (-80% decline!)
- Topic coherence: 0.5492 → 0.0275 (-95% decline!)
- Balance score: 0.661 → 0.170 (-74% decline!)
- Large outlier cluster: 139 emails (23.2% of dataset)

✅ **Minor Improvements:**
- Silhouette score: 0.0195 → 0.0398 (+104% improvement)
- Auto-detected 5 meaningful topics + outliers
- Successfully integrated BERTopic pipeline

**Verdict:** BERTopic implementation is technically sound, but results are poor. Likely causes: (1) different dataset (602 vs 1,147 emails), (2) parameter tuning needed, (3) evaluation metrics mismatch, or (4) smaller dataset insufficient for BERTopic.

---

## Phase 2 Implementation Details

### Tasks Completed

#### ✅ Task 2.1: Install BERTopic Dependencies
- **Packages installed:** bertopic 0.17.3, umap-learn 0.5.9, hdbscan 0.8.40
- **Status:** Successful
- **Time:** ~2 minutes

#### ✅ Task 2.2: Create BERTopic Discovery Script
- **File created:** `scripts/categorization/bertopic_discovery.py` (309 lines)
- **Features:**
  - Extends CategoryDiscovery class (inheritance pattern)
  - Uses pre-computed embeddings (no re-embedding cost)
  - UMAP dimensionality reduction (5D)
  - HDBSCAN density-based clustering
  - c-TF-IDF keyword extraction (trigrams)
  - LLM category naming (reused from Phase 1)
- **Status:** Successful

#### ✅ Task 2.3: Run BERTopic Discovery
- **Dataset:** Primo_List_2 (602 emails, 599 after filtering)
- **Parameters:**
  - min_cluster_size: 30
  - UMAP: n_neighbors=15, n_components=5, metric=cosine
  - HDBSCAN: metric=euclidean, cluster_selection_method=eom
  - Vectorizer: ngram_range=(1,3), min_df=5
- **Execution time:** ~10 seconds
- **Status:** Successful

---

## Detailed Metrics Comparison

### Phase 1 (K-means) vs Phase 2 (BERTopic)

| Metric | Phase 1 (K-means) | Phase 2 (BERTopic) | Change | Status |
|--------|-------------------|-------------------|--------|--------|
| **Dataset** | 1,147 emails | 602 emails | -47% | ⚠️ Different |
| **Silhouette Score** | 0.0195 | 0.0398 | +104% | ✅ Improved |
| **Balance Score** | 0.661 | 0.170 | -74% | ❌ Worse |
| **Topic Coherence** | 0.5492 | 0.0275 | -95% | ❌ Much worse |
| **Overall Quality** | **0.371** | **0.075** | **-80%** | **❌ Worse** |
| **Categories** | 5 | 5 + 1 outlier | Same | ✅ |
| **Outliers** | 0 (0%) | 139 (23.2%) | +139 | ⚠️ High |

**Note:** Different datasets make direct comparison challenging. Phase 1 used Primo_List (1,147 emails), Phase 2 used Primo_List_2 (602 emails).

---

## BERTopic Discovered Categories

### Category Distribution

| Category | Count | Percentage | Coherence |
|----------|-------|------------|-----------|
| Library System Integration | 280 | 46.7% | 0.0800 |
| Outliers | 139 | 23.2% | N/A |
| Call for Proposals | 59 | 9.8% | 0.0160 |
| User Interface Feedback | 58 | 9.7% | 0.0240 |
| Technical Support | 32 | 5.3% | 0.0020 |
| Search Results | 31 | 5.2% | 0.0156 |

**Observations:**
- Large imbalance: Largest category (280) is 9x larger than smallest (31)
- High outlier rate (23.2%) suggests many emails don't fit discovered topics
- Very low coherence scores across all categories

### Sample Category Analysis

**Library System Integration (280 emails, 46.7%)**
- **BERTopic keywords:** hover, external, display, saved, facet, labels
- **Top senders:** Stacey van Groll (71), Susan Barber (52)
- **Interpretation:** General Primo UI/configuration questions
- **Issue:** Too broad - captures multiple distinct topics

**User Interface Feedback (58 emails, 9.7%)**
- **BERTopic keywords:** facet, behaviour, search, history, saved
- **Interpretation:** Facet behavior in NDE UI
- **Issue:** Very specific - single discussion thread

**Outliers (139 emails, 23.2%)**
- **Issue:** Too many emails classified as outliers
- **Likely cause:** min_cluster_size=30 too strict for small dataset

---

## Root Cause Analysis

### Why Did BERTopic Perform Poorly?

#### 1. **Different Dataset** ⚠️ Primary Suspect
- **Phase 1:** 1,147 emails (Primo_List)
- **Phase 2:** 602 emails (Primo_List_2)
- **Impact:** BERTopic needs larger datasets (research recommends 500+ *per topic*)
- **Conclusion:** 602 emails insufficient for meaningful clustering

#### 2. **High Outlier Rate** (23.2%)
- **Cause:** min_cluster_size=30 too strict
- **Effect:** Many emails labeled as noise instead of being clustered
- **Fix:** Lower min_cluster_size to 15-20 for small datasets

#### 3. **Imbalanced Cluster Sizes**
- **Largest:** 280 emails (46.7%)
- **Smallest:** 31 emails (5.2%)
- **Ratio:** 9:1
- **Effect:** Balance score dropped from 0.661 to 0.170

#### 4. **Evaluation Metric Mismatch** ⚠️ Investigation Needed
- **Coherence dropped:** 0.5492 → 0.0275 (-95%)
- **Possible cause:** Evaluation script uses different coherence calculation than BERTopic's c-TF-IDF
- **Question:** Are we measuring the right thing?

#### 5. **Parameter Tuning** ⚠️ Not Optimized
- Used default/recommended parameters from plan
- No hyperparameter tuning performed
- UMAP/HDBSCAN have many parameters that affect clustering

---

## Comparison: Expected vs Actual

### Research-Based Expectations (from Phase 2 Plan)

| Metric | Phase 1 | Expected Phase 2 | Actual Phase 2 | Status |
|--------|---------|------------------|----------------|--------|
| Silhouette | 0.15 | 0.35 (+133%) | 0.0398 | ⚠️ Much worse than expected |
| Balance | 0.70 | 0.80 (+14%) | 0.170 | ❌ Much worse |
| Coherence | 0.40 | 0.60 (+50%) | 0.0275 | ❌ Catastrophic decline |
| Overall | 0.42 | 0.58 (+38%) | 0.075 | ❌ Catastrophic decline |

**Conclusion:** Results contradict research literature expectations for BERTopic performance.

---

## Technical Implementation Quality

### ✅ What Worked

1. **Clean Code Architecture**
   - BERTopicCategoryDiscovery extends CategoryDiscovery
   - Reused LLM naming, TF-IDF rules, person name filtering
   - Minimal code duplication

2. **Embedding Reuse**
   - Used pre-computed embeddings (no re-embedding cost)
   - UMAP dimensionality reduction worked correctly
   - HDBSCAN clustering executed successfully

3. **LLM Category Naming**
   - Generated meaningful names: "Library System Integration", "User Interface Feedback"
   - Avoided duplicates
   - Cost: ~$0.005 (5 categories × $0.001)

4. **BERTopic Integration**
   - Library installed correctly
   - No version conflicts
   - Script runs end-to-end without errors

### ⚠️ What Needs Investigation

1. **Coherence Metric**
   - Dramatic decline (-95%) suggests evaluation issue
   - Need to verify coherence calculation methodology
   - Consider using BERTopic's native coherence scores

2. **Dataset Size**
   - 602 emails may be too small for BERTopic
   - Research recommends 500+ emails *per topic* (we have 5 topics = 2,500+ emails needed)
   - Phase 1 used 1,147 emails

3. **Parameter Tuning**
   - No hyperparameter search performed
   - min_cluster_size=30 may be too strict
   - UMAP/HDBSCAN defaults may not suit email data

---

## Recommendations

### Immediate Actions (Before Phase 3)

#### Option A: Re-run Phase 2 on Original Dataset ⭐ RECOMMENDED
**Rationale:** Fair comparison requires same dataset

```bash
# 1. Verify Primo_List has embeddings
# 2. Re-run Phase 1 on Primo_List for baseline
# 3. Re-run Phase 2 on Primo_List (apples-to-apples comparison)
# 4. Compare results
```

**Expected outcome:** Better understanding of whether BERTopic truly underperforms

#### Option B: Tune BERTopic Parameters
**Rationale:** Default parameters may not be optimal

```python
# Try these adjustments:
min_cluster_size=15  # Lower threshold (was 30)
min_samples=5        # HDBSCAN parameter
n_neighbors=30       # UMAP parameter (was 15)
ngram_range=(1, 2)   # Bigrams only (was 1-3 trigrams)
```

**Expected outcome:** Reduced outliers, better balance

#### Option C: Investigate Coherence Metric ⭐ CRITICAL
**Rationale:** -95% decline is suspicious

```python
# 1. Check how coherence is calculated in evaluate_categories.py
# 2. Compare with BERTopic's native c-TF-IDF coherence
# 3. Verify TF-IDF keywords are being used correctly
```

**Expected outcome:** Understand if metric is valid

---

### Long-Term Recommendations

1. **Standardize Evaluation**
   - Create test dataset with ground truth labels
   - Human evaluation: Sample 50 emails, measure accuracy
   - Use multiple coherence metrics (NPMI, UCI, etc.)

2. **Dataset Requirements**
   - BERTopic works best with 1,000+ emails
   - For production, wait until project has sufficient data
   - Consider combining multiple projects for discovery

3. **Hybrid Approach**
   - Use BERTopic for discovery (find topics)
   - Use K-means for production (faster, simpler)
   - Leverage BERTopic's c-TF-IDF for better keywords

---

## Files Created/Modified

### Created
1. **`scripts/categorization/bertopic_discovery.py`** (309 lines)
   - BERTopicCategoryDiscovery class
   - UMAP + HDBSCAN + c-TF-IDF pipeline
   - Extends Phase 1 architecture

2. **`data/categories/discovered_categories_bertopic.json`** (520 KB)
   - 6 categories (5 + outliers)
   - BERTopic keywords and centroids
   - Discovery metadata

3. **`docs/future/EMAIL_CATEGORIZATION_PHASE2_RESULTS.md`** (this document)

### Modified
1. **`pyproject.toml`** - Added bertopic, umap-learn, hdbscan dependencies
2. **`scripts/categorization/evaluate_categories.py`** - Updated to use Primo_List_2

---

## Lessons Learned

### What We Learned

1. **Dataset Size Matters**
   - BERTopic needs substantially more data than K-means
   - 602 emails is borderline too small
   - Outlier detection is more aggressive than expected

2. **Parameter Sensitivity**
   - min_cluster_size has major impact on outlier rate
   - Default parameters don't work for all datasets
   - Need dataset-specific tuning

3. **Evaluation is Critical**
   - Different coherence metrics give different results
   - Need multiple metrics to validate quality
   - Same dataset is essential for fair comparison

4. **Implementation vs Performance**
   - Technical implementation: ✅ Success
   - Performance results: ❌ Disappointing
   - These are separate concerns

### Research Hypothesis to Test

**Hypothesis:** BERTopic excels with large, diverse corpora (10,000+ documents) but underperforms on small, focused datasets (< 1,000 documents).

**Test:** Run both K-means and BERTopic on datasets of varying sizes:
- Small (500-1,000 emails)
- Medium (1,000-3,000 emails)
- Large (3,000+ emails)

**Expected finding:** BERTopic advantage emerges at larger scales.

---

## Decision Point: Proceed to Phase 3?

### Arguments AGAINST Proceeding

1. **Poor Results:** Overall quality declined 80%
2. **Different Dataset:** Can't validate improvement without fair comparison
3. **No Baseline:** Don't know if this is BERTopic's fault or data issue
4. **Unclear ROI:** Phase 3 builds on Phase 2 - risky foundation

### Arguments FOR Proceeding

1. **Phase 3 is Different:** Thread analysis, temporal topics (not clustering)
2. **Phase 1 Sufficient:** Quality of 0.371 may be good enough for MVP
3. **Time Investment:** Phase 2 infrastructure is built
4. **Learning Value:** Phase 3 features are valuable regardless

### **RECOMMENDATION: PAUSE Phase 3** ⚠️

**Recommended Next Steps:**
1. Re-run Phase 2 on Primo_List (same dataset as Phase 1)
2. Investigate coherence metric calculation
3. Try parameter tuning (lower min_cluster_size)
4. Document fair comparison results
5. **THEN** decide: Phase 3 or pivot to production with Phase 1

**Estimated time:** 2-3 hours to complete investigation

---

## Performance Metrics

### Execution Time
- **Dependency installation:** ~2 minutes
- **Script creation:** ~5 minutes (manual coding)
- **Discovery execution:** ~10 seconds
- **Evaluation:** ~5 seconds
- **Total Phase 2 time:** ~15-20 minutes (excluding investigation)

### Cost Analysis
- **Dependency installation:** $0
- **BERTopic clustering:** $0 (local computation)
- **LLM category naming:** ~$0.005 (5 categories)
- **Embedding reuse:** $0
- **Total cost:** ~$0.005

---

## Next Session Priorities

1. **CRITICAL:** Re-run on same dataset for fair comparison
2. **IMPORTANT:** Investigate coherence metric decline
3. **MEDIUM:** Try parameter tuning (min_cluster_size)
4. **LOW:** Document future feature request (runtime model selection)

---

**Document Version:** 1.0
**Last Updated:** 2025-11-23
**Status:** Phase 2 Complete - Investigation Required Before Phase 3
**Next Review:** After re-running on Primo_List dataset
