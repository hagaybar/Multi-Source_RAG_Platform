# Query Testing & Reporting Guide

This guide explains how to systematically test your RAG pipeline with detailed reporting.

## 🚀 Quick Start

```bash
# 1. Edit your test questions
nano sample_test_questions.txt

# 2. Run the test suite
PYTHONPATH=/home/hagaybar/projects/Multi-Source_RAG_Platform poetry run python run_test_queries.py \
    --questions sample_test_questions.txt \
    --output test_results/

# 3. Review the reports
cd test_results/
ls -lh
```

## 📊 What You Get

### Individual Query Reports

For each query, you get a detailed markdown report (`run_*_report.md`) containing:

1. **Query & Answer** - The question and the LLM's response
2. **Decision Flow** - How the system understood and processed the query:
   - Intent detection (temporal_query, factual_lookup, etc.)
   - Confidence scores
   - Strategy selection (multi_aspect, late_fusion, etc.)
   - Temporal constraints applied
3. **Retrieval Results** - What was retrieved:
   - Number of chunks
   - Date range of retrieved emails
   - Top senders
   - Similarity scores
   - Preview of top 5 chunks (subject, date, sender, text preview)
4. **Context Assembly** - How the prompt was built:
   - Prompt length
   - Chunk inclusion verification
5. **Performance Metrics** - Timing for each step
6. **Quality Assessment** - Automatic issue detection:
   - Zero results
   - Low confidence
   - Short answers
   - "No information" responses

### Aggregate Summary Report

The `SUMMARY_REPORT.md` provides cross-query analysis:

1. **Execution Overview**
   - Total chunks retrieved across all queries
   - Average chunks per query
   - Queries with zero results
   - Answer quality metrics

2. **Intent Detection Analysis**
   - Intent distribution across queries
   - Average confidence scores
   - Detection method breakdown (pattern vs LLM)

3. **Strategy Analysis**
   - Which strategies were used
   - Strategy effectiveness (avg chunks per strategy)

4. **Performance Analysis**
   - Response times

5. **Problem Areas**
   - Queries that had issues (with specific problems listed)
   - Common failure patterns

6. **Successful Queries**
   - Examples of well-performing queries

7. **Recommendations**
   - Actionable suggestions based on results

8. **Complete Query List**
   - Table with all queries and key metrics

### Machine-Readable Data

`summary_data.json` contains all the raw data in JSON format for programmatic analysis.

## 📝 Writing Test Questions

Edit `sample_test_questions.txt` (or create your own file):

```text
# Comments start with #
# One question per line

# Temporal queries (test the recent/latest fix)
What is the latest information about Citation Styles?
Recent discussions about Chicago bibliography

# Factual queries
How do I configure facets in Primo?
What is the Research Assistant feature?

# Topic-based queries
What are the pressing issues with Primo?
Issues with loading or performance
```

**Tips:**
- Mix different query types (temporal, factual, topic-based)
- Include queries you know should work
- Include queries that might be challenging
- Test edge cases (very specific, very broad)

## 🔍 Analyzing Results

### Step 1: Review Individual Reports

Start with queries that had issues (marked with ⚠️ in the quick results):

```bash
# Find reports with issues
grep -l "⚠️ Issues Detected" test_results/*_report.md
```

Read each report to understand:
- Was the intent detected correctly?
- Were relevant chunks retrieved?
- Why did the LLM say "no information"?

### Step 2: Review Summary Report

Open `SUMMARY_REPORT.md` and focus on:
- **Problem Areas section** - Common issues across queries
- **Recommendations section** - What to fix
- **Intent Detection Analysis** - Are queries being understood correctly?

### Step 3: Iterate

Based on findings:
- Adjust temporal filter settings (if too many zero results)
- Improve intent detection patterns
- Add more test queries to cover edge cases
- Refine retrieval parameters

## 📈 Success Criteria

A healthy system should have:
- ✅ < 20% zero-result queries
- ✅ > 60% average intent confidence
- ✅ > 80% queries with answers > 50 chars
- ✅ < 30% "no information" responses

## 🛠️ Advanced Usage

### Test Specific Project

```bash
python run_test_queries.py \
    --project data/projects/MyProject \
    --questions my_questions.txt \
    --output my_results/
```

### Re-analyze Existing Runs

If you already ran queries through the UI and want to generate reports:

```python
from pathlib import Path
from scripts.analysis.query_report_generator import generate_reports, AggregateReportGenerator

# Generate report for a specific run
run_dir = Path("data/projects/Primo_List_2/logs/runs/run_20251124_113000")
output_dir = Path("manual_reports")
metadata = generate_reports(run_dir, output_dir)

# Generate summary for multiple runs
reports = []
for run_dir in Path("data/projects/Primo_List_2/logs/runs").iterdir():
    if run_dir.is_dir():
        try:
            metadata = generate_reports(run_dir, output_dir)
            reports.append(metadata)
        except:
            pass

generator = AggregateReportGenerator(reports)
summary = generator.generate_summary_report()

with open(output_dir / "SUMMARY_REPORT.md", 'w') as f:
    f.write(summary)
```

## 📚 Example Report Structure

```
test_results/
├── SUMMARY_REPORT.md                 # Aggregate analysis
├── summary_data.json                 # Machine-readable data
├── run_20251124_113000_report.md    # Individual report for query 1
├── run_20251124_113030_report.md    # Individual report for query 2
├── run_20251124_113100_report.md    # Individual report for query 3
└── ...
```

## 💡 Pro Tips

1. **Start Small**: Test 3-5 queries first to verify the system works
2. **Compare Before/After**: Run tests before and after making changes
3. **Version Your Questions**: Keep different versions of test questions for different scenarios
4. **Track Metrics Over Time**: Save reports with dates to track improvements
5. **Share Results**: Reports are markdown - easy to share with team

## 🐛 Troubleshooting

**"No module named 'scripts.analysis'"**
- Make sure you're running with PYTHONPATH set correctly
- Ensure scripts/analysis/__init__.py exists

**"No run directories found"**
- Make sure queries actually ran through the pipeline
- Check that project logs directory exists

**"KeyError in report generation"**
- Some old runs may be missing artifact files
- Only works with runs that have complete artifacts

## 📞 Need Help?

If you encounter issues or want to customize the reports, the report generator code is in:
- `scripts/analysis/query_report_generator.py` - Report generation logic
- `run_test_queries.py` - Test runner script

Feel free to modify the report format to suit your needs!
