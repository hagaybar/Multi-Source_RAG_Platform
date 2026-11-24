# Email RAG Evaluation Framework

This directory contains tools for systematically evaluating the email RAG system's performance.

## Overview

The evaluation framework consists of:

1. **Test Suite** (`tests/fixtures/evaluation_test_suite.yaml`) - 30 test cases covering diverse query types
2. **Evaluation Runner** (`run_evaluation_suite.py`) - Automated test execution and reporting
3. **Pattern Analyzer** (`analyze_patterns.py`) - Identifies common failure patterns across tests
4. **Manual Review Template** - Generated for human quality assessment

## Quick Start

### Run Complete Evaluation

```bash
# From project root
cd tests/evaluation
python run_evaluation_suite.py
```

This will:
- Run all 30 test cases from the test suite
- Generate detailed results in `results/eval_<timestamp>/`
- Create three output files:
  - `results.json` - Full machine-readable results
  - `report.md` - Human-readable summary report
  - `manual_review.md` - Template for manual quality assessment

### Analyze Results

```bash
# Analyze patterns from latest evaluation run
python analyze_patterns.py results/eval_<timestamp>/

# This generates:
# - analysis.json - Machine-readable pattern analysis
# - analysis.md - Human-readable pattern report with recommendations
```

## Test Suite Structure

The test suite (`tests/fixtures/evaluation_test_suite.yaml`) contains 30 test cases across 8 categories:

### Categories

1. **Exploratory Queries (5 tests)** - User doesn't know what they want but knows HOW to find it
   - Examples: "What are the most pressing issues?", "Who are the most active participants?"
   - Success criteria: Thread-level retrieval, full context, heuristic ranking

2. **Specific Queries (5 tests)** - User knows exactly what they want
   - Examples: "How do I configure facets?", "What did John say about NDE?"
   - Success criteria: Precise retrieval, direct answers

3. **Vague Queries (3 tests)** - User knows but expresses poorly
   - Examples: "What's happening with Primo?", "Tell me about issues"
   - Success criteria: Query reformulation, multiple sub-queries

4. **Implicit Knowledge Queries (5 tests)** - Requires domain knowledge
   - Examples: "EU01 login problems", "CSL repo location"
   - Success criteria: Domain knowledge injection, entity recognition

5. **Thread-Based Queries (3 tests)** - Full thread context needed
   - Examples: "Summarize the discussion about X", "Show all messages from thread Y"
   - Success criteria: Complete thread retrieval, chronological order

6. **Edge Cases (4 tests)** - Unusual or problematic inputs
   - Examples: Non-existent sender, no data queries, gibberish
   - Success criteria: Graceful degradation, informative errors

7. **Temporal Queries (3 tests)** - Time-based filtering
   - Examples: "Issues from past month", "Recent updates", "Compare periods"
   - Success criteria: Temporal constraint extraction, date filtering

8. **Aggregation Queries (3 tests)** - Summary across multiple items
   - Examples: "All bugs mentioned", "Opinions on feature X"
   - Success criteria: Comprehensive coverage, structured aggregation

### Test Case Format

Each test case includes:

```yaml
- id: EXP-01
  query: "What are the most pressing issues discussed in the past 90 days?"
  category: exploratory
  subcategory: hot_topics

  expected_behavior: |
    Should identify the 3-5 discussion threads with the most messages
    in the past 90 days that are about issues/problems/bugs.
    Should return FULL context from these threads (all messages).

  success_criteria:
    chunk_count: ">= 30"
    unique_threads: "<= 5"
    avg_chunks_per_thread: ">= 6"
    answer_length: ">= 500 chars"

  manual_review_points:
    - Are the retrieved threads actually the LONGEST threads about issues?
    - Does the answer summarize multiple distinct issues?
    - Are the issues actually "pressing" (unresolved, active discussion)?

  domain_knowledge_required: ["issue severity keywords", "thread metadata"]

  implicit_assumptions:
    - "Pressing" = high message volume + unresolved status
    - "Issues" = problems/bugs, not enhancements/questions
```

## Evaluation Metrics

### Automated Criteria

The evaluation runner automatically checks:

- **chunk_count** - Number of retrieved chunks (supports `>=`, `<=`, `>`, `<`, `==`)
- **unique_threads** - Number of distinct email threads
- **avg_chunks_per_thread** - Average chunks per thread (measures fragmentation)
- **answer_length** - Length of generated answer in characters
- **has_reformulation** - Whether query was reformulated (true/false)
- **intent_detected** - Detected intent type (e.g., "aggregation_query")
- **contains_sender** - Whether specific sender's emails are present
- **time_range_matches** - Whether temporal constraint was correctly extracted

### Manual Review Points

Each test includes human-reviewable aspects:

- Relevance of retrieved content
- Quality of generated answer
- Accuracy of domain knowledge application
- Handling of implicit assumptions
- Overall user satisfaction

## Output Files

### results.json

Full machine-readable results including:
```json
{
  "run_id": "eval_20251125_120000",
  "timestamp": "20251125_120000",
  "total_tests": 30,
  "test_results": [
    {
      "test_id": "EXP-01",
      "query": "...",
      "evaluation": {
        "criteria_results": [...],
        "overall_pass": true,
        "manual_review_points": [...]
      },
      "retrieval_output": {
        "chunks": [...],
        "intent": {...},
        "reformulated_queries": [...]
      }
    }
  ],
  "summary": {
    "passed": 25,
    "failed": 5,
    "pass_rate": "83.3%"
  }
}
```

### report.md

Human-readable summary with:
- Overall pass/fail statistics
- Per-category performance
- Individual test results with criteria pass/fail
- Quick visual feedback (✓/✗ indicators)

### manual_review.md

Template for human review with:
- Manual review questions for each test
- Quality rating scale (1-5)
- Space for notes and observations
- Implicit assumptions to verify

### analysis.json / analysis.md

Pattern analysis including:
- Most common failure criteria
- Performance by category and intent type
- Reformulation effectiveness analysis
- Chunk distribution statistics
- Prioritized recommendations

## Workflow

### 1. Initial Evaluation

```bash
# Run full test suite
python run_evaluation_suite.py

# Review automated results
cat results/eval_<timestamp>/report.md
```

### 2. Pattern Analysis

```bash
# Identify common issues
python analyze_patterns.py results/eval_<timestamp>/

# Review recommendations
cat results/eval_<timestamp>/analysis.md
```

### 3. Manual Review

```bash
# Open manual review template
code results/eval_<timestamp>/manual_review.md

# For each test:
# - Read the query and expected behavior
# - Review retrieved chunks (in results.json)
# - Read the generated answer
# - Answer manual review questions
# - Rate quality 1-5
# - Note any issues
```

### 4. Fix & Iterate

Based on analysis:
1. Prioritize fixes by frequency and impact
2. Implement targeted improvements
3. Re-run evaluation
4. Compare results to measure progress

## Example Usage

### Complete Evaluation Session

```bash
# 1. Run evaluation
python run_evaluation_suite.py
# Output: Results saved to tests/evaluation/results/eval_20251125_120000

# 2. Analyze patterns
python analyze_patterns.py results/eval_20251125_120000
# Output:
#   - Priority 1: Fix 'unique_threads' criterion - failed in 8 tests
#   - Priority 2: Investigate 'exploratory' category - only 40% pass rate

# 3. Review specific failures
grep -A 20 "EXP-01" results/eval_20251125_120000/report.md

# 4. Manual review
code results/eval_20251125_120000/manual_review.md
```

### Tracking Progress Over Time

```bash
# Run baseline evaluation
python run_evaluation_suite.py
# Save results_dir: results/eval_baseline

# Make improvements to system
# ...

# Run new evaluation
python run_evaluation_suite.py
# Save results_dir: results/eval_after_fix

# Compare
python analyze_patterns.py results/eval_baseline
python analyze_patterns.py results/eval_after_fix

# Check if pass rate improved
```

## Adding New Test Cases

To add new test cases to the suite:

1. Edit `tests/fixtures/evaluation_test_suite.yaml`
2. Add new test case following the format above
3. Assign unique ID (category prefix + number)
4. Define success criteria using supported criterion types
5. Add manual review points for human assessment
6. Run evaluation to verify test works

## Interpreting Results

### High Pass Rate (>80%)

System is performing well. Focus on:
- Manual review for quality assessment
- Edge cases and implicit knowledge
- User satisfaction in practice

### Medium Pass Rate (50-80%)

System has room for improvement. Check:
- Which categories are failing most?
- Which criteria are most problematic?
- Pattern analysis recommendations

### Low Pass Rate (<50%)

System needs significant work. Priorities:
1. Fix error cases first
2. Address most common failure criterion
3. Focus on worst-performing category
4. Consider architectural changes

## Architecture Integration

The evaluation framework tests the complete pipeline:

1. **Query Intelligence (Phase 6)** - Analysis, reformulation, domain knowledge
2. **Intent Detection** - 7 intent types
3. **Strategy Selection** - Specialized retrievers
4. **Retrieval Execution** - Vector search
5. **Reranking (Phase 5)** - Quality assessment
6. **Context Assembly** - Deduplication, token management
7. **Answer Generation** - LLM prompting

Each test exercises the full stack, allowing identification of issues at any stage.

## Known Limitations

1. **Automated criteria are incomplete** - Many quality aspects require human judgment
2. **Test suite is snapshot** - May not cover all real-world query variations
3. **Success thresholds are initial estimates** - May need calibration based on data
4. **No performance/latency testing** - Focuses on accuracy, not speed
5. **Assumes Primo_List_2 project** - Hardcoded project path

## Future Enhancements

- [ ] Automated comparison between evaluation runs
- [ ] Performance/latency benchmarking
- [ ] Configurable project selection
- [ ] Web UI for manual review
- [ ] Integration with CI/CD
- [ ] Synthetic test data generation
- [ ] User feedback collection

## Troubleshooting

### "Results file not found"
- Check that `run_evaluation_suite.py` completed successfully
- Verify the results directory path

### "Project path not found"
- Update `project_path` in `run_evaluation_suite.py:main()`
- Ensure `data/projects/Primo_List_2` exists

### Tests taking too long
- Reduce test suite size temporarily
- Run specific test IDs only (TODO: add CLI filtering)
- Use faster embedder model

### All tests failing with errors
- Check API keys (OpenAI)
- Verify FAISS indices exist
- Review app.log for detailed errors

## Support

For questions or issues:
1. Check pattern analysis recommendations
2. Review manual_review.md for specific test insights
3. Check `data/projects/Primo_List_2/logs/` for detailed logs
4. Consult main project documentation

---

**Last Updated:** 2025-01-25
**Version:** 1.0
