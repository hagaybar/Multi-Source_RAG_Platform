#!/usr/bin/env python3
"""
Run a batch of test queries and generate detailed reports.

Usage:
    python run_test_queries.py --questions test_questions.txt --output test_results/

This will:
1. Run each query through the pipeline
2. Generate individual detailed report for each query
3. Generate aggregate summary report with analysis

Reports include:
- Query & answer
- Decision flow (intent detection, strategy selection)
- Retrieval results (chunks, dates, senders, similarity scores)
- Context assembly (prompt construction)
- Performance metrics
- Quality assessment
- Aggregate analysis across all queries
"""
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from scripts.core.project_manager import ProjectManager
from scripts.pipeline.runner import PipelineRunner
from scripts.analysis.query_report_generator import (
    generate_reports,
    AggregateReportGenerator
)

def run_test_query(project, config, query, output_dir):
    """Run a single test query and generate detailed report."""
    print(f"\n{'='*80}")
    print(f"Query: {query}")
    print('='*80)

    # Create a fresh runner for each query to get unique run_id
    runner = PipelineRunner(project, config)

    # Run pipeline
    try:
        # Execute retrieve step
        print("\n🔍 Running retrieval...")
        for msg in runner.step_retrieve(query=query, top_k=None):
            print(f"  {msg}")

        # Execute ask step
        print("\n🧠 Generating answer...")
        for msg in runner.step_ask(query=query):
            print(f"  {msg}")

        # Get latest run directory
        runs_dir = project.root_dir / "logs" / "runs"
        latest_run = max(runs_dir.iterdir(), key=lambda p: p.stat().st_mtime)

        print(f"\n📊 Generating detailed report...")
        # Generate individual report using the reporting system
        report_metadata = generate_reports(latest_run, output_dir)

        print(f"✅ Individual report saved: {report_metadata['report_file']}")
        return report_metadata

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description="Run batch test queries")
    parser.add_argument("--project", type=str, default="data/projects/Primo_List_2",
                       help="Project path")
    parser.add_argument("--questions", type=str, required=True,
                       help="File containing questions (one per line)")
    parser.add_argument("--output", type=str, default="test_results",
                       help="Output directory for results")

    args = parser.parse_args()

    # Setup
    project_path = Path(args.project)
    questions_file = Path(args.questions)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load questions
    if not questions_file.exists():
        print(f"❌ Questions file not found: {questions_file}")
        return 1

    with open(questions_file) as f:
        questions = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    print(f"📋 Loaded {len(questions)} test questions")
    print(f"📁 Project: {project_path}")
    print(f"📂 Output: {output_dir}")

    # Load project and config (runner will be created per-query)
    project = ProjectManager(project_path)
    config = project.config  # Get config from project

    # Run tests
    all_reports = []
    for i, question in enumerate(questions, 1):
        print(f"\n\n{'#'*80}")
        print(f"# TEST {i}/{len(questions)}")
        print(f"{'#'*80}")

        report_metadata = run_test_query(project, config, question, output_dir)
        if report_metadata:
            all_reports.append(report_metadata)

    # Generate aggregate summary report
    print(f"\n\n{'='*80}")
    print(f"📊 Generating aggregate summary report...")
    print('='*80)

    summary_generator = AggregateReportGenerator(all_reports)
    summary_md = summary_generator.generate_summary_report()

    summary_file = output_dir / "SUMMARY_REPORT.md"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(summary_md)

    print(f"✅ Summary report saved: {summary_file}")

    # Also save machine-readable JSON
    summary_json = output_dir / "summary_data.json"
    with open(summary_json, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_tests": len(questions),
            "successful_tests": len(all_reports),
            "reports": all_reports
        }, f, indent=2)

    print(f"\n\n{'='*80}")
    print(f"✅ TESTING COMPLETE!")
    print('='*80)
    print(f"\n📊 Reports Generated:")
    print(f"  • Summary Report: {summary_file}")
    print(f"  • Data Summary: {summary_json}")
    print(f"  • Individual Reports: {len(all_reports)} files in {output_dir}/")
    print(f"\n📁 Output Directory: {output_dir}")
    print('='*80)

    # Print quick summary
    print("\n📊 QUICK RESULTS:")
    for r in all_reports:
        status = "✅" if r['chunks_retrieved'] > 0 and r['answer_length'] > 50 else "⚠️"
        print(f"  {status} {r['query'][:60]}")
        print(f"     Intent: {r['intent']} | Chunks: {r['chunks_retrieved']} | Answer: {r['answer_length']} chars")

    # Print summary stats
    total_chunks = sum(r['chunks_retrieved'] for r in all_reports)
    zero_results = sum(1 for r in all_reports if r['chunks_retrieved'] == 0)
    avg_chunks = total_chunks / len(all_reports) if all_reports else 0

    print(f"\n📈 Statistics:")
    print(f"  • Total queries: {len(all_reports)}")
    print(f"  • Zero results: {zero_results}")
    print(f"  • Avg chunks/query: {avg_chunks:.1f}")
    print(f"  • Total chunks retrieved: {total_chunks}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
