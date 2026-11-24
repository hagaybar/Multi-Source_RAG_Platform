#!/usr/bin/env python3
"""
Evaluation Script for Email RAG System
Runs test cases from evaluation_test_suite.yaml and generates detailed reports.
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import yaml
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.core.project_manager import ProjectManager
from scripts.agents.email_orchestrator import EmailOrchestratorAgent
from scripts.utils.logger import LoggerManager


class EvaluationRunner:
    """Runs evaluation test suite and generates reports."""

    def __init__(self, project_path: Path, test_suite_path: Path):
        self.project_path = project_path
        self.test_suite_path = test_suite_path
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"eval_{self.timestamp}"

        # Create results directory
        self.results_dir = project_root / "tests" / "evaluation" / "results" / self.run_id
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize logger
        self.logger = LoggerManager.get_logger("evaluation")

        # Load project and orchestrator
        self.project = ProjectManager(project_path)
        self.orchestrator = EmailOrchestratorAgent(self.project, run_id=self.run_id)

        # Load test suite
        with open(test_suite_path, 'r') as f:
            self.test_suite = yaml.safe_load(f)

        self.logger.info(f"Evaluation runner initialized", extra={
            "run_id": self.run_id,
            "test_count": len(self.test_suite['test_cases']),
            "results_dir": str(self.results_dir)
        })

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all test cases and collect results."""
        results = {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "project": str(self.project_path),
            "total_tests": len(self.test_suite['test_cases']),
            "test_results": []
        }

        for i, test_case in enumerate(self.test_suite['test_cases'], 1):
            self.logger.info(f"Running test {i}/{results['total_tests']}: {test_case['id']}")
            print(f"\n{'='*80}")
            print(f"Test {i}/{results['total_tests']}: {test_case['id']} - {test_case['query']}")
            print(f"{'='*80}")

            test_result = self.run_single_test(test_case)
            results['test_results'].append(test_result)

            # Print immediate feedback
            self._print_test_summary(test_result)

        # Calculate summary statistics
        results['summary'] = self._calculate_summary(results['test_results'])

        # Save results
        self._save_results(results)

        return results

    def run_single_test(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single test case and evaluate results."""
        test_id = test_case['id']
        query = test_case['query']

        result = {
            "test_id": test_id,
            "query": query,
            "category": test_case['category'],
            "subcategory": test_case.get('subcategory', 'N/A'),
            "timestamp": datetime.now().isoformat(),
            "error": None,
            "retrieval_output": None,
            "evaluation": {
                "criteria_results": [],
                "overall_pass": False,
                "needs_manual_review": True
            }
        }

        try:
            # Run retrieval
            retrieval_output = self.orchestrator.retrieve(query, top_k=50, max_tokens=8000)
            result['retrieval_output'] = self._serialize_retrieval_output(retrieval_output)

            # Evaluate against success criteria
            if 'success_criteria' in test_case:
                criteria_results = self._evaluate_criteria(
                    test_case['success_criteria'],
                    retrieval_output
                )
                result['evaluation']['criteria_results'] = criteria_results
                result['evaluation']['overall_pass'] = all(
                    cr['passed'] for cr in criteria_results
                )

            # Add manual review prompts
            if 'manual_review_points' in test_case:
                result['evaluation']['manual_review_points'] = test_case['manual_review_points']

            # Add domain knowledge requirements
            if 'domain_knowledge_required' in test_case:
                result['evaluation']['domain_knowledge_required'] = test_case['domain_knowledge_required']

            # Add implicit assumptions
            if 'implicit_assumptions' in test_case:
                result['evaluation']['implicit_assumptions'] = test_case['implicit_assumptions']

        except Exception as e:
            self.logger.error(f"Test {test_id} failed with error", extra={
                "test_id": test_id,
                "error": str(e)
            }, exc_info=True)
            result['error'] = str(e)

        return result

    def _evaluate_criteria(self, criteria: Dict[str, Any], retrieval_output: Dict) -> List[Dict[str, Any]]:
        """Evaluate success criteria against retrieval output."""
        criteria_results = []

        for criterion_name, expected_value in criteria.items():
            criterion_result = {
                "criterion": criterion_name,
                "expected": expected_value,
                "actual": None,
                "passed": False,
                "notes": ""
            }

            try:
                if criterion_name == "chunk_count":
                    actual = len(retrieval_output.get('chunks', []))
                    criterion_result['actual'] = actual
                    criterion_result['passed'] = self._compare_value(actual, expected_value)

                elif criterion_name == "unique_threads":
                    thread_ids = set()
                    for chunk in retrieval_output.get('chunks', []):
                        thread_id = chunk.meta.get('thread_id') or chunk.meta.get('subject')
                        if thread_id:
                            thread_ids.add(thread_id)
                    actual = len(thread_ids)
                    criterion_result['actual'] = actual
                    criterion_result['passed'] = self._compare_value(actual, expected_value)

                elif criterion_name == "avg_chunks_per_thread":
                    chunks = retrieval_output.get('chunks', [])
                    if not chunks:
                        actual = 0
                    else:
                        thread_ids = {}
                        for chunk in chunks:
                            thread_id = chunk.meta.get('thread_id') or chunk.meta.get('subject')
                            if thread_id:
                                thread_ids[thread_id] = thread_ids.get(thread_id, 0) + 1
                        actual = sum(thread_ids.values()) / len(thread_ids) if thread_ids else 0
                    criterion_result['actual'] = round(actual, 2)
                    criterion_result['passed'] = self._compare_value(actual, expected_value)

                elif criterion_name == "answer_length":
                    answer = retrieval_output.get('final_answer', '')
                    actual = len(answer)
                    criterion_result['actual'] = f"{actual} chars"
                    criterion_result['passed'] = self._compare_value(actual, expected_value)

                elif criterion_name == "has_reformulation":
                    actual = len(retrieval_output.get('reformulated_queries', [])) > 0
                    expected = expected_value.lower() == 'true'
                    criterion_result['actual'] = actual
                    criterion_result['passed'] = actual == expected

                elif criterion_name == "intent_detected":
                    actual = retrieval_output.get('intent', {}).get('primary_intent')
                    criterion_result['actual'] = actual
                    criterion_result['passed'] = actual == expected_value

                elif criterion_name == "contains_sender":
                    chunks = retrieval_output.get('chunks', [])
                    expected_sender = expected_value
                    found = any(
                        chunk.meta.get('from', '').lower() == expected_sender.lower()
                        for chunk in chunks
                    )
                    criterion_result['actual'] = found
                    criterion_result['passed'] = found

                elif criterion_name == "time_range_matches":
                    # Check if temporal constraint matches expected
                    intent = retrieval_output.get('intent', {})
                    temporal_constraint = intent.get('metadata', {}).get('temporal_constraint', {})
                    actual = temporal_constraint.get('days_back')
                    criterion_result['actual'] = actual

                    # Parse expected value (e.g., "30 days")
                    if isinstance(expected_value, str) and 'days' in expected_value.lower():
                        expected_days = int(expected_value.split()[0])
                        criterion_result['passed'] = actual == expected_days
                    else:
                        criterion_result['passed'] = False
                        criterion_result['notes'] = f"Cannot parse expected value: {expected_value}"

                else:
                    criterion_result['notes'] = f"Unknown criterion: {criterion_name}"

            except Exception as e:
                criterion_result['notes'] = f"Error evaluating: {str(e)}"

            criteria_results.append(criterion_result)

        return criteria_results

    def _compare_value(self, actual: Any, expected_str: str) -> bool:
        """Compare actual value against expected string (supports operators like >=, <=, etc.)."""
        expected_str = str(expected_str).strip()

        # Handle comparison operators
        if expected_str.startswith('>='):
            threshold = self._parse_number(expected_str[2:])
            return actual >= threshold
        elif expected_str.startswith('<='):
            threshold = self._parse_number(expected_str[2:])
            return actual <= threshold
        elif expected_str.startswith('>'):
            threshold = self._parse_number(expected_str[1:])
            return actual > threshold
        elif expected_str.startswith('<'):
            threshold = self._parse_number(expected_str[1:])
            return actual < threshold
        elif expected_str.startswith('=='):
            threshold = self._parse_number(expected_str[2:])
            return actual == threshold
        else:
            # Exact match
            threshold = self._parse_number(expected_str)
            return actual == threshold

    def _parse_number(self, value_str: str) -> float:
        """Parse number from string, handling units like 'chars'."""
        value_str = value_str.strip()
        # Remove units
        value_str = value_str.replace('chars', '').replace('tokens', '').strip()
        return float(value_str)

    def _serialize_retrieval_output(self, output: Dict) -> Dict:
        """Serialize retrieval output for JSON storage."""
        serialized = {}

        # Copy simple fields
        for key in ['query', 'reformulated_queries', 'intent', 'strategies',
                    'total_chunks', 'final_answer', 'domain_context']:
            if key in output:
                serialized[key] = output[key]

        # Serialize chunks
        if 'chunks' in output:
            serialized['chunks'] = [
                {
                    'id': chunk.id,
                    'text': chunk.text[:500] + '...' if len(chunk.text) > 500 else chunk.text,
                    'token_count': chunk.token_count,
                    'meta': chunk.meta
                }
                for chunk in output['chunks']
            ]

        # Serialize query analysis
        if 'query_analysis' in output:
            qa = output['query_analysis']
            serialized['query_analysis'] = {
                'is_specific': qa.is_specific,
                'is_vague': qa.is_vague,
                'clarity_score': qa.clarity_score,
                'has_temporal_constraint': qa.has_temporal_constraint,
                'entities': qa.entities,
                'issues': qa.issues
            }

        return serialized

    def _calculate_summary(self, test_results: List[Dict]) -> Dict[str, Any]:
        """Calculate summary statistics from test results."""
        total = len(test_results)
        passed = sum(1 for r in test_results if r['evaluation']['overall_pass'])
        failed = sum(1 for r in test_results if not r['evaluation']['overall_pass'])
        errors = sum(1 for r in test_results if r['error'] is not None)

        # Group by category
        by_category = {}
        for result in test_results:
            category = result['category']
            if category not in by_category:
                by_category[category] = {'total': 0, 'passed': 0, 'failed': 0}
            by_category[category]['total'] += 1
            if result['evaluation']['overall_pass']:
                by_category[category]['passed'] += 1
            else:
                by_category[category]['failed'] += 1

        return {
            'total_tests': total,
            'passed': passed,
            'failed': failed,
            'errors': errors,
            'pass_rate': f"{(passed/total*100):.1f}%" if total > 0 else "N/A",
            'by_category': by_category
        }

    def _print_test_summary(self, result: Dict):
        """Print immediate feedback for a test."""
        status = "✓ PASS" if result['evaluation']['overall_pass'] else "✗ FAIL"
        if result['error']:
            status = "✗ ERROR"

        print(f"\nStatus: {status}")

        if result['error']:
            print(f"Error: {result['error']}")
        else:
            # Print criteria results
            for cr in result['evaluation']['criteria_results']:
                passed_str = "✓" if cr['passed'] else "✗"
                print(f"  {passed_str} {cr['criterion']}: expected {cr['expected']}, got {cr['actual']}")
                if cr['notes']:
                    print(f"      Note: {cr['notes']}")

    def _save_results(self, results: Dict):
        """Save results to files."""
        # Save full results as JSON
        results_file = self.results_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        # Generate human-readable report
        report_file = self.results_dir / "report.md"
        self._generate_markdown_report(results, report_file)

        # Save manual review template
        review_file = self.results_dir / "manual_review.md"
        self._generate_manual_review_template(results, review_file)

        self.logger.info("Results saved", extra={
            "results_file": str(results_file),
            "report_file": str(report_file),
            "review_file": str(review_file)
        })

        print(f"\n{'='*80}")
        print(f"Results saved to: {self.results_dir}")
        print(f"  - Full results: results.json")
        print(f"  - Human report: report.md")
        print(f"  - Manual review: manual_review.md")
        print(f"{'='*80}\n")

    def _generate_markdown_report(self, results: Dict, output_path: Path):
        """Generate human-readable markdown report."""
        lines = [
            f"# Evaluation Report - {results['run_id']}",
            f"",
            f"**Timestamp:** {results['timestamp']}",
            f"**Project:** {results['project']}",
            f"",
            f"## Summary",
            f"",
            f"- **Total Tests:** {results['summary']['total_tests']}",
            f"- **Passed:** {results['summary']['passed']}",
            f"- **Failed:** {results['summary']['failed']}",
            f"- **Errors:** {results['summary']['errors']}",
            f"- **Pass Rate:** {results['summary']['pass_rate']}",
            f"",
            f"### By Category",
            f""
        ]

        for category, stats in results['summary']['by_category'].items():
            pass_rate = f"{(stats['passed']/stats['total']*100):.1f}%" if stats['total'] > 0 else "N/A"
            lines.append(f"- **{category}:** {stats['passed']}/{stats['total']} passed ({pass_rate})")

        lines.extend([
            f"",
            f"---",
            f"",
            f"## Test Results",
            f""
        ])

        for result in results['test_results']:
            status = "✓ PASS" if result['evaluation']['overall_pass'] else "✗ FAIL"
            if result['error']:
                status = "✗ ERROR"

            lines.extend([
                f"### {result['test_id']}: {status}",
                f"",
                f"**Query:** {result['query']}",
                f"**Category:** {result['category']} / {result['subcategory']}",
                f""
            ])

            if result['error']:
                lines.append(f"**Error:** {result['error']}")
                lines.append("")
            else:
                lines.append("**Criteria Results:**")
                lines.append("")
                for cr in result['evaluation']['criteria_results']:
                    passed_str = "✓" if cr['passed'] else "✗"
                    lines.append(f"- {passed_str} **{cr['criterion']}**: expected `{cr['expected']}`, got `{cr['actual']}`")
                    if cr['notes']:
                        lines.append(f"  - Note: {cr['notes']}")
                lines.append("")

        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))

    def _generate_manual_review_template(self, results: Dict, output_path: Path):
        """Generate manual review template."""
        lines = [
            f"# Manual Review Template - {results['run_id']}",
            f"",
            f"Use this template to manually review test results that require human judgment.",
            f"",
            f"**Instructions:**",
            f"1. For each test below, review the retrieved chunks and generated answer",
            f"2. Answer the manual review questions",
            f"3. Rate the overall quality (1-5 scale)",
            f"4. Note any issues or insights",
            f"",
            f"---",
            f""
        ]

        for i, result in enumerate(results['test_results'], 1):
            lines.extend([
                f"## Test {i}: {result['test_id']}",
                f"",
                f"**Query:** {result['query']}",
                f"**Category:** {result['category']} / {result['subcategory']}",
                f"**Automated Pass:** {'Yes' if result['evaluation']['overall_pass'] else 'No'}",
                f""
            ])

            if 'manual_review_points' in result['evaluation']:
                lines.append("**Manual Review Questions:**")
                lines.append("")
                for point in result['evaluation']['manual_review_points']:
                    lines.append(f"- [ ] {point}")
                    lines.append(f"  - Answer: _[Your assessment here]_")
                lines.append("")

            if 'implicit_assumptions' in result['evaluation']:
                lines.append("**Implicit Assumptions:**")
                lines.append("")
                for assumption in result['evaluation']['implicit_assumptions']:
                    lines.append(f"- {assumption}")
                lines.append("")

            lines.extend([
                f"**Quality Rating (1-5):** _____",
                f"",
                f"**Issues Found:**",
                f"- _[List any problems]_",
                f"",
                f"**Notes:**",
                f"- _[Additional observations]_",
                f"",
                f"---",
                f""
            ])

        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))


def main():
    """Main entry point."""
    # Configuration
    project_path = Path("data/projects/Primo_List_2")
    test_suite_path = Path("tests/fixtures/evaluation_test_suite.yaml")

    # Validate paths
    if not project_path.exists():
        print(f"Error: Project path not found: {project_path}")
        sys.exit(1)

    if not test_suite_path.exists():
        print(f"Error: Test suite not found: {test_suite_path}")
        sys.exit(1)

    # Run evaluation
    print(f"\n{'='*80}")
    print(f"Starting Evaluation Suite")
    print(f"{'='*80}\n")
    print(f"Project: {project_path}")
    print(f"Test Suite: {test_suite_path}")
    print(f"\nThis will run {len(yaml.safe_load(open(test_suite_path))['test_cases'])} test cases...\n")

    runner = EvaluationRunner(project_path, test_suite_path)
    results = runner.run_all_tests()

    # Print final summary
    print(f"\n{'='*80}")
    print(f"Evaluation Complete!")
    print(f"{'='*80}")
    print(f"Total Tests: {results['summary']['total_tests']}")
    print(f"Passed: {results['summary']['passed']}")
    print(f"Failed: {results['summary']['failed']}")
    print(f"Errors: {results['summary']['errors']}")
    print(f"Pass Rate: {results['summary']['pass_rate']}")
    print(f"\nResults directory: {runner.results_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
