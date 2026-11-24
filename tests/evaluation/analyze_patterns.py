#!/usr/bin/env python3
"""
Pattern Analysis Tool for Evaluation Results
Analyzes evaluation results to identify common failure patterns.
"""

import sys
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any
import argparse


class PatternAnalyzer:
    """Analyzes evaluation results to identify patterns."""

    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.results_file = results_dir / "results.json"

        if not self.results_file.exists():
            raise FileNotFoundError(f"Results file not found: {self.results_file}")

        with open(self.results_file, 'r') as f:
            self.results = json.load(f)

    def analyze_all(self) -> Dict[str, Any]:
        """Run all analysis patterns."""
        analysis = {
            "run_id": self.results['run_id'],
            "summary": self.results['summary'],
            "failure_patterns": self.analyze_failure_patterns(),
            "criteria_analysis": self.analyze_criteria_performance(),
            "category_analysis": self.analyze_by_category(),
            "intent_analysis": self.analyze_intent_patterns(),
            "reformulation_analysis": self.analyze_reformulation_patterns(),
            "chunk_distribution": self.analyze_chunk_distribution(),
            "recommendations": []
        }

        # Generate recommendations based on patterns
        analysis['recommendations'] = self.generate_recommendations(analysis)

        return analysis

    def analyze_failure_patterns(self) -> Dict[str, Any]:
        """Identify common failure patterns."""
        patterns = {
            "total_failures": 0,
            "failure_categories": defaultdict(list),
            "most_common_failures": [],
            "error_cases": []
        }

        for result in self.results['test_results']:
            if result['error']:
                patterns['error_cases'].append({
                    'test_id': result['test_id'],
                    'query': result['query'],
                    'error': result['error']
                })
                patterns['total_failures'] += 1
            elif not result['evaluation']['overall_pass']:
                patterns['total_failures'] += 1
                patterns['failure_categories'][result['category']].append({
                    'test_id': result['test_id'],
                    'query': result['query'],
                    'failed_criteria': [
                        cr['criterion'] for cr in result['evaluation']['criteria_results']
                        if not cr['passed']
                    ]
                })

        # Find most common failure types
        criterion_failures = defaultdict(int)
        for category, failures in patterns['failure_categories'].items():
            for failure in failures:
                for criterion in failure['failed_criteria']:
                    criterion_failures[criterion] += 1

        patterns['most_common_failures'] = sorted(
            criterion_failures.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return patterns

    def analyze_criteria_performance(self) -> Dict[str, Any]:
        """Analyze performance of each success criterion."""
        criteria_stats = defaultdict(lambda: {'total': 0, 'passed': 0, 'failed': 0})

        for result in self.results['test_results']:
            if result['error']:
                continue

            for cr in result['evaluation']['criteria_results']:
                criterion = cr['criterion']
                criteria_stats[criterion]['total'] += 1
                if cr['passed']:
                    criteria_stats[criterion]['passed'] += 1
                else:
                    criteria_stats[criterion]['failed'] += 1

        # Calculate pass rates
        for criterion, stats in criteria_stats.items():
            if stats['total'] > 0:
                stats['pass_rate'] = f"{(stats['passed'] / stats['total'] * 100):.1f}%"

        return dict(criteria_stats)

    def analyze_by_category(self) -> Dict[str, Any]:
        """Deep dive into performance by category."""
        category_analysis = {}

        # Group tests by category
        by_category = defaultdict(list)
        for result in self.results['test_results']:
            by_category[result['category']].append(result)

        for category, tests in by_category.items():
            analysis = {
                'total': len(tests),
                'passed': sum(1 for t in tests if t['evaluation']['overall_pass']),
                'failed': sum(1 for t in tests if not t['evaluation']['overall_pass']),
                'errors': sum(1 for t in tests if t['error']),
                'common_issues': []
            }

            # Find common failure criteria for this category
            criterion_failures = defaultdict(int)
            for test in tests:
                if not test['evaluation']['overall_pass'] and not test['error']:
                    for cr in test['evaluation']['criteria_results']:
                        if not cr['passed']:
                            criterion_failures[cr['criterion']] += 1

            analysis['common_issues'] = sorted(
                criterion_failures.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]  # Top 3 issues

            category_analysis[category] = analysis

        return category_analysis

    def analyze_intent_patterns(self) -> Dict[str, Any]:
        """Analyze intent detection patterns."""
        intent_stats = defaultdict(lambda: {'count': 0, 'passed': 0, 'failed': 0})

        for result in self.results['test_results']:
            if result['error'] or not result['retrieval_output']:
                continue

            intent_data = result['retrieval_output'].get('intent', {})
            primary_intent = intent_data.get('primary_intent', 'unknown')

            intent_stats[primary_intent]['count'] += 1
            if result['evaluation']['overall_pass']:
                intent_stats[primary_intent]['passed'] += 1
            else:
                intent_stats[primary_intent]['failed'] += 1

        # Calculate pass rates
        for intent, stats in intent_stats.items():
            if stats['count'] > 0:
                stats['pass_rate'] = f"{(stats['passed'] / stats['count'] * 100):.1f}%"

        return dict(intent_stats)

    def analyze_reformulation_patterns(self) -> Dict[str, Any]:
        """Analyze query reformulation patterns."""
        reformulation_stats = {
            'total_reformulated': 0,
            'not_reformulated': 0,
            'reformulation_by_category': defaultdict(int),
            'pass_rate_reformulated': None,
            'pass_rate_not_reformulated': None
        }

        reformulated_passed = 0
        not_reformulated_passed = 0

        for result in self.results['test_results']:
            if result['error'] or not result['retrieval_output']:
                continue

            reformulated_queries = result['retrieval_output'].get('reformulated_queries', [])
            was_reformulated = len(reformulated_queries) > 0

            if was_reformulated:
                reformulation_stats['total_reformulated'] += 1
                reformulation_stats['reformulation_by_category'][result['category']] += 1
                if result['evaluation']['overall_pass']:
                    reformulated_passed += 1
            else:
                reformulation_stats['not_reformulated'] += 1
                if result['evaluation']['overall_pass']:
                    not_reformulated_passed += 1

        # Calculate pass rates
        if reformulation_stats['total_reformulated'] > 0:
            reformulation_stats['pass_rate_reformulated'] = \
                f"{(reformulated_passed / reformulation_stats['total_reformulated'] * 100):.1f}%"

        if reformulation_stats['not_reformulated'] > 0:
            reformulation_stats['pass_rate_not_reformulated'] = \
                f"{(not_reformulated_passed / reformulation_stats['not_reformulated'] * 100):.1f}%"

        return reformulation_stats

    def analyze_chunk_distribution(self) -> Dict[str, Any]:
        """Analyze distribution of retrieved chunks."""
        chunk_stats = {
            'avg_chunks': 0,
            'min_chunks': float('inf'),
            'max_chunks': 0,
            'avg_unique_threads': 0,
            'chunk_distribution_by_category': {}
        }

        total_chunks = 0
        total_tests = 0
        total_unique_threads = 0

        by_category = defaultdict(lambda: {'chunks': [], 'threads': []})

        for result in self.results['test_results']:
            if result['error'] or not result['retrieval_output']:
                continue

            chunks = result['retrieval_output'].get('chunks', [])
            num_chunks = len(chunks)

            # Count unique threads
            thread_ids = set()
            for chunk in chunks:
                thread_id = chunk['meta'].get('thread_id') or chunk['meta'].get('subject')
                if thread_id:
                    thread_ids.add(thread_id)

            num_threads = len(thread_ids)

            total_chunks += num_chunks
            total_tests += 1
            total_unique_threads += num_threads

            chunk_stats['min_chunks'] = min(chunk_stats['min_chunks'], num_chunks)
            chunk_stats['max_chunks'] = max(chunk_stats['max_chunks'], num_chunks)

            # By category
            by_category[result['category']]['chunks'].append(num_chunks)
            by_category[result['category']]['threads'].append(num_threads)

        if total_tests > 0:
            chunk_stats['avg_chunks'] = round(total_chunks / total_tests, 1)
            chunk_stats['avg_unique_threads'] = round(total_unique_threads / total_tests, 1)

        # Calculate per-category stats
        for category, data in by_category.items():
            chunk_stats['chunk_distribution_by_category'][category] = {
                'avg_chunks': round(sum(data['chunks']) / len(data['chunks']), 1) if data['chunks'] else 0,
                'avg_threads': round(sum(data['threads']) / len(data['threads']), 1) if data['threads'] else 0
            }

        return chunk_stats

    def generate_recommendations(self, analysis: Dict) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []

        # Check failure patterns
        if analysis['failure_patterns']['total_failures'] > 0:
            most_common = analysis['failure_patterns']['most_common_failures']
            if most_common:
                top_failure = most_common[0]
                recommendations.append(
                    f"Priority 1: Fix '{top_failure[0]}' criterion - failed in {top_failure[1]} tests"
                )

        # Check category performance
        worst_category = None
        worst_pass_rate = 100.0
        for category, stats in analysis['category_analysis'].items():
            if stats['total'] > 0:
                pass_rate = (stats['passed'] / stats['total']) * 100
                if pass_rate < worst_pass_rate:
                    worst_pass_rate = pass_rate
                    worst_category = category

        if worst_category and worst_pass_rate < 50:
            recommendations.append(
                f"Priority 2: Investigate '{worst_category}' category - only {worst_pass_rate:.1f}% pass rate"
            )

        # Check chunk distribution issues
        chunk_dist = analysis['chunk_distribution']
        if chunk_dist['avg_unique_threads'] < 3:
            recommendations.append(
                f"Consider: Average unique threads ({chunk_dist['avg_unique_threads']}) is low - "
                "may indicate fragmented results"
            )

        # Check reformulation effectiveness
        ref_analysis = analysis['reformulation_analysis']
        if ref_analysis['pass_rate_reformulated'] and ref_analysis['pass_rate_not_reformulated']:
            ref_rate = float(ref_analysis['pass_rate_reformulated'].rstrip('%'))
            non_ref_rate = float(ref_analysis['pass_rate_not_reformulated'].rstrip('%'))

            if ref_rate < non_ref_rate - 10:
                recommendations.append(
                    f"Warning: Reformulated queries have lower pass rate ({ref_rate:.1f}%) "
                    f"than non-reformulated ({non_ref_rate:.1f}%) - reformulation may need tuning"
                )

        # Check for errors
        if analysis['failure_patterns']['error_cases']:
            recommendations.append(
                f"Critical: {len(analysis['failure_patterns']['error_cases'])} tests had errors - "
                "fix these first"
            )

        if not recommendations:
            recommendations.append("Great! No critical issues found. Focus on manual review for quality.")

        return recommendations

    def save_analysis(self, output_path: Path):
        """Save analysis to file."""
        analysis = self.analyze_all()

        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)

        # Also create markdown report
        md_path = output_path.with_suffix('.md')
        self._create_markdown_report(analysis, md_path)

        print(f"Analysis saved to:")
        print(f"  JSON: {output_path}")
        print(f"  Markdown: {md_path}")

    def _create_markdown_report(self, analysis: Dict, output_path: Path):
        """Create human-readable markdown report."""
        lines = [
            f"# Pattern Analysis Report - {analysis['run_id']}",
            "",
            "## Summary",
            "",
            f"- **Total Tests:** {analysis['summary']['total_tests']}",
            f"- **Passed:** {analysis['summary']['passed']}",
            f"- **Failed:** {analysis['summary']['failed']}",
            f"- **Pass Rate:** {analysis['summary']['pass_rate']}",
            "",
            "---",
            "",
            "## Recommendations",
            ""
        ]

        for i, rec in enumerate(analysis['recommendations'], 1):
            lines.append(f"{i}. {rec}")

        lines.extend([
            "",
            "---",
            "",
            "## Failure Patterns",
            "",
            f"**Total Failures:** {analysis['failure_patterns']['total_failures']}",
            ""
        ])

        if analysis['failure_patterns']['error_cases']:
            lines.append("### Error Cases")
            lines.append("")
            for error in analysis['failure_patterns']['error_cases']:
                lines.append(f"- **{error['test_id']}**: {error['query']}")
                lines.append(f"  - Error: `{error['error']}`")
            lines.append("")

        if analysis['failure_patterns']['most_common_failures']:
            lines.append("### Most Common Failure Criteria")
            lines.append("")
            for criterion, count in analysis['failure_patterns']['most_common_failures'][:5]:
                lines.append(f"- **{criterion}**: {count} failures")
            lines.append("")

        lines.extend([
            "---",
            "",
            "## Performance by Category",
            ""
        ])

        for category, stats in analysis['category_analysis'].items():
            pass_rate = f"{(stats['passed']/stats['total']*100):.1f}%" if stats['total'] > 0 else "N/A"
            lines.append(f"### {category}")
            lines.append("")
            lines.append(f"- **Total:** {stats['total']}")
            lines.append(f"- **Passed:** {stats['passed']}")
            lines.append(f"- **Failed:** {stats['failed']}")
            lines.append(f"- **Pass Rate:** {pass_rate}")

            if stats['common_issues']:
                lines.append("- **Common Issues:**")
                for issue, count in stats['common_issues']:
                    lines.append(f"  - {issue}: {count} failures")
            lines.append("")

        lines.extend([
            "---",
            "",
            "## Criteria Performance",
            ""
        ])

        for criterion, stats in sorted(
            analysis['criteria_analysis'].items(),
            key=lambda x: float(x[1]['pass_rate'].rstrip('%')) if 'pass_rate' in x[1] else 0
        ):
            lines.append(f"- **{criterion}**: {stats['pass_rate']} ({stats['passed']}/{stats['total']})")

        lines.extend([
            "",
            "---",
            "",
            "## Intent Detection Analysis",
            ""
        ])

        for intent, stats in analysis['intent_analysis'].items():
            lines.append(f"- **{intent}**: {stats['count']} queries, {stats['pass_rate']} pass rate")

        lines.extend([
            "",
            "---",
            "",
            "## Query Reformulation",
            "",
            f"- **Reformulated Queries:** {analysis['reformulation_analysis']['total_reformulated']}",
            f"- **Not Reformulated:** {analysis['reformulation_analysis']['not_reformulated']}",
            f"- **Pass Rate (Reformulated):** {analysis['reformulation_analysis']['pass_rate_reformulated']}",
            f"- **Pass Rate (Not Reformulated):** {analysis['reformulation_analysis']['pass_rate_not_reformulated']}",
            "",
            "---",
            "",
            "## Chunk Distribution",
            "",
            f"- **Average Chunks per Query:** {analysis['chunk_distribution']['avg_chunks']}",
            f"- **Min Chunks:** {analysis['chunk_distribution']['min_chunks']}",
            f"- **Max Chunks:** {analysis['chunk_distribution']['max_chunks']}",
            f"- **Average Unique Threads:** {analysis['chunk_distribution']['avg_unique_threads']}",
            "",
            "### By Category",
            ""
        ])

        for category, stats in analysis['chunk_distribution']['chunk_distribution_by_category'].items():
            lines.append(f"- **{category}**: {stats['avg_chunks']} chunks, {stats['avg_threads']} threads avg")

        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Analyze evaluation results for patterns')
    parser.add_argument('results_dir', type=Path, help='Path to evaluation results directory')
    parser.add_argument('-o', '--output', type=Path, help='Output path for analysis (default: analysis.json in results dir)')

    args = parser.parse_args()

    if not args.results_dir.exists():
        print(f"Error: Results directory not found: {args.results_dir}")
        sys.exit(1)

    output_path = args.output or (args.results_dir / "analysis.json")

    print(f"Analyzing results from: {args.results_dir}")
    print(f"Output will be saved to: {output_path}")

    analyzer = PatternAnalyzer(args.results_dir)
    analyzer.save_analysis(output_path)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
