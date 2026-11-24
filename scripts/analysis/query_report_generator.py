#!/usr/bin/env python3
"""
Generate detailed reports for test query executions.

Creates:
1. Individual query reports (markdown format)
2. Aggregate summary report
"""
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from collections import Counter, defaultdict


class QueryReportGenerator:
    """Generate detailed reports for query executions."""

    def __init__(self, run_dir: Path):
        """
        Initialize with a run directory.

        Args:
            run_dir: Path to the run directory (e.g., logs/runs/run_20251124_113000/)
        """
        self.run_dir = Path(run_dir)
        self.run_id = run_dir.name

        # Load artifacts
        self.metadata = self._load_json(run_dir / "run_metadata.json")
        self.chunks = self._load_jsonl(run_dir / "retrieved_chunks.jsonl")
        self.prompt = self._load_text(run_dir / "llm_prompt.txt")
        self.response = self._load_text(run_dir / "llm_response.txt")
        self.app_log = self._load_jsonl(run_dir / "app.log")

    def _load_json(self, path: Path) -> Optional[Dict]:
        """Load JSON file."""
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)

    def _load_jsonl(self, path: Path) -> List[Dict]:
        """Load JSONL file."""
        if not path.exists():
            return []
        lines = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    try:
                        lines.append(json.loads(line))
                    except:
                        pass
        return lines

    def _load_text(self, path: Path) -> Optional[str]:
        """Load text file."""
        if not path.exists():
            return None
        with open(path, encoding='utf-8') as f:
            return f.read()

    def generate_individual_report(self) -> str:
        """
        Generate detailed individual query report.

        Returns:
            Markdown formatted report
        """
        lines = []

        # Header
        lines.append(f"# Query Execution Report")
        lines.append(f"**Run ID:** `{self.run_id}`")
        lines.append(f"**Timestamp:** {self.metadata.get('timestamp', 'N/A')}")
        lines.append("")

        # Section 1: Query & Answer
        lines.append("## 📝 Query & Answer")
        lines.append("")
        lines.append(f"**Query:**")
        lines.append(f"> {self.metadata.get('query', 'N/A')}")
        lines.append("")
        lines.append(f"**Answer:**")
        lines.append(f"> {self.response or 'No answer generated'}")
        lines.append("")
        lines.append(f"**Answer Length:** {len(self.response or '')} characters")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Section 2: Decision Flow
        lines.append("## 🧠 Decision Flow")
        lines.append("")

        intent_info = self.metadata.get('intent', {})
        lines.append("### 1️⃣ Intent Detection")
        lines.append(f"- **Primary Intent:** `{intent_info.get('primary_intent', 'N/A')}`")
        lines.append(f"- **Confidence:** {intent_info.get('confidence', 0):.2f}")
        lines.append(f"- **Detection Method:** {intent_info.get('detection_method', 'N/A')}")
        lines.append("")

        # Intent metadata
        intent_meta = intent_info.get('metadata', {})
        if intent_meta:
            lines.append("**Intent Metadata:**")
            if 'time_range' in intent_meta:
                lines.append(f"- Time Range: `{intent_meta['time_range']}`")
            if 'temporal_constraint' in intent_meta:
                tc = intent_meta['temporal_constraint']
                lines.append(f"- Temporal Constraint: {tc.get('keyword', 'N/A')} ({tc.get('days_back', 'N/A')} days)")
            if 'topic_keywords' in intent_meta:
                keywords = ', '.join(intent_meta['topic_keywords'][:5])
                lines.append(f"- Topic Keywords: {keywords}")
            if 'sender' in intent_meta:
                lines.append(f"- Sender Filter: {intent_meta['sender']}")
            lines.append("")

        lines.append("### 2️⃣ Strategy Selection")
        lines.append(f"- **Strategy:** `{self.metadata.get('strategy', 'N/A')}`")
        lines.append(f"- **Orchestrator:** `{self.metadata.get('orchestrator', 'standard')}`")
        lines.append(f"- **Top-K Requested:** {self.metadata.get('top_k', 'N/A')}")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Section 3: Retrieval Results
        lines.append("## 🔍 Retrieval Results")
        lines.append("")
        lines.append(f"**Total Chunks Retrieved:** {len(self.chunks)}")
        lines.append("")

        if self.chunks:
            # Date distribution
            dates = [c.get('meta', {}).get('date', '') for c in self.chunks if c.get('meta', {}).get('date')]
            if dates:
                dates_sorted = sorted(dates, reverse=True)
                lines.append(f"**Date Range:**")
                lines.append(f"- Most Recent: {dates_sorted[0]}")
                lines.append(f"- Oldest: {dates_sorted[-1]}")
                lines.append("")

            # Sender distribution
            senders = [c.get('meta', {}).get('sender_name', 'Unknown') for c in self.chunks]
            sender_counts = Counter(senders).most_common(5)
            lines.append("**Top Senders:**")
            for sender, count in sender_counts:
                lines.append(f"- {sender}: {count} chunks")
            lines.append("")

            # Similarity scores
            similarities = [c.get('similarity', 0) for c in self.chunks]
            if similarities:
                lines.append("**Similarity Scores:**")
                lines.append(f"- Best: {max(similarities):.4f}")
                lines.append(f"- Worst: {min(similarities):.4f}")
                lines.append(f"- Average: {sum(similarities)/len(similarities):.4f}")
                lines.append("")

            # Top 5 chunks
            lines.append("### 📧 Top Retrieved Chunks")
            lines.append("")
            for i, chunk in enumerate(self.chunks[:5], 1):
                meta = chunk.get('meta', {})
                subject = meta.get('subject', 'No subject')
                date = meta.get('date', 'No date')
                sender = meta.get('sender_name', 'Unknown')
                similarity = chunk.get('similarity', 0)
                text_preview = chunk.get('text', '')[:150].replace('\n', ' ')

                lines.append(f"**{i}. {subject[:80]}**")
                lines.append(f"- Date: {date}")
                lines.append(f"- Sender: {sender}")
                lines.append(f"- Similarity: {similarity:.4f}")
                lines.append(f"- Preview: *{text_preview}...*")
                lines.append("")
        else:
            lines.append("⚠️ **No chunks retrieved!**")
            lines.append("")

        lines.append("---")
        lines.append("")

        # Section 4: Context Assembly
        lines.append("## 🏗️ Context Assembly")
        lines.append("")

        if self.prompt:
            lines.append(f"**Prompt Length:** {len(self.prompt)} characters (~{len(self.prompt.split())} words)")
            lines.append("")
            lines.append("**Prompt Structure:**")
            prompt_lines = self.prompt.split('\n')
            lines.append(f"- First line: {prompt_lines[0][:100]}...")
            lines.append(f"- Last line: {prompt_lines[-1][:100]}...")
            lines.append(f"- Total lines: {len(prompt_lines)}")
            lines.append("")

            # Check if chunks are in prompt
            if self.chunks and len(self.chunks) > 0:
                first_chunk_preview = self.chunks[0].get('text', '')[:50]
                if first_chunk_preview in self.prompt:
                    lines.append("✅ Chunks are included in the prompt")
                else:
                    lines.append("⚠️ Chunks may not be properly included in prompt")
                lines.append("")
        else:
            lines.append("⚠️ No prompt file found")
            lines.append("")

        lines.append("---")
        lines.append("")

        # Section 5: Performance
        lines.append("## ⚡ Performance Metrics")
        lines.append("")

        # Parse timing from app.log
        timing = {}
        for entry in self.app_log:
            if 'elapsed_ms' in entry:
                step = entry.get('message', 'unknown')
                timing[step] = entry['elapsed_ms']

        if timing:
            lines.append("**Execution Time:**")
            for step, ms in timing.items():
                lines.append(f"- {step}: {ms}ms ({ms/1000:.2f}s)")
            lines.append("")
            total_ms = sum(timing.values())
            lines.append(f"**Total Time:** {total_ms}ms ({total_ms/1000:.2f}s)")
        else:
            lines.append("No timing data available")

        lines.append("")
        lines.append("---")
        lines.append("")

        # Section 6: Quality Assessment
        lines.append("## 📊 Quality Assessment")
        lines.append("")

        # Automatic checks
        issues = []
        successes = []

        if len(self.chunks) == 0:
            issues.append("No chunks retrieved")
        elif len(self.chunks) < 3:
            issues.append(f"Low chunk count ({len(self.chunks)})")
        else:
            successes.append(f"Retrieved {len(self.chunks)} chunks")

        if not self.response or len(self.response) < 50:
            issues.append("Very short answer (< 50 chars)")
        elif "no information" in self.response.lower() or "cannot provide" in self.response.lower():
            issues.append("LLM could not find information in context")
        else:
            successes.append("Generated substantial answer")

        if intent_info.get('confidence', 0) < 0.5:
            issues.append(f"Low intent confidence ({intent_info.get('confidence', 0):.2f})")
        else:
            successes.append(f"High intent confidence ({intent_info.get('confidence', 0):.2f})")

        lines.append("**✅ Successes:**")
        for s in successes:
            lines.append(f"- {s}")
        lines.append("")

        if issues:
            lines.append("**⚠️ Issues Detected:**")
            for issue in issues:
                lines.append(f"- {issue}")
            lines.append("")

        lines.append("**Manual Review Needed:**")
        lines.append("- [ ] Answer is factually correct")
        lines.append("- [ ] Answer addresses the query fully")
        lines.append("- [ ] Retrieved chunks are relevant")
        lines.append("- [ ] No important information missed")
        lines.append("")

        return '\n'.join(lines)


class AggregateReportGenerator:
    """Generate aggregate summary report across multiple query executions."""

    def __init__(self, reports: List[Dict]):
        """
        Initialize with individual report data.

        Args:
            reports: List of report dictionaries with metadata
        """
        self.reports = reports

    def generate_summary_report(self) -> str:
        """
        Generate aggregate summary report.

        Returns:
            Markdown formatted summary
        """
        lines = []

        # Header
        lines.append(f"# Test Execution Summary Report")
        lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Total Queries:** {len(self.reports)}")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Section 1: Overview
        lines.append("## 📊 Execution Overview")
        lines.append("")

        # Success metrics
        total_chunks = sum(r.get('chunks_retrieved', 0) for r in self.reports)
        avg_chunks = total_chunks / len(self.reports) if self.reports else 0
        zero_results = sum(1 for r in self.reports if r.get('chunks_retrieved', 0) == 0)

        lines.append(f"**Retrieval Statistics:**")
        lines.append(f"- Total chunks retrieved: {total_chunks}")
        lines.append(f"- Average chunks per query: {avg_chunks:.1f}")
        if self.reports:
            lines.append(f"- Queries with 0 results: {zero_results} ({zero_results/len(self.reports)*100:.1f}%)")
        else:
            lines.append(f"- Queries with 0 results: N/A (no successful reports)")
        lines.append("")

        # Answer quality
        avg_answer_len = sum(r.get('answer_length', 0) for r in self.reports) / len(self.reports) if self.reports else 0
        short_answers = sum(1 for r in self.reports if r.get('answer_length', 0) < 50)
        no_info_answers = sum(1 for r in self.reports if 'no information' in r.get('answer', '').lower())

        lines.append(f"**Answer Quality:**")
        lines.append(f"- Average answer length: {avg_answer_len:.0f} characters")
        lines.append(f"- Very short answers (< 50 chars): {short_answers}")
        lines.append(f"- 'No information' responses: {no_info_answers}")
        lines.append("")

        lines.append("---")
        lines.append("")

        # Section 2: Intent Detection Analysis
        lines.append("## 🧠 Intent Detection Analysis")
        lines.append("")

        intent_counts = Counter(r.get('intent', 'unknown') for r in self.reports)
        avg_confidence = sum(r.get('confidence', 0) for r in self.reports) / len(self.reports) if self.reports else 0

        lines.append(f"**Intent Distribution:**")
        for intent, count in intent_counts.most_common():
            pct = count / len(self.reports) * 100
            lines.append(f"- `{intent}`: {count} queries ({pct:.1f}%)")
        lines.append("")

        lines.append(f"**Average Confidence:** {avg_confidence:.2f}")
        lines.append("")

        # Detection method breakdown
        detection_methods = Counter(r.get('detection_method', 'unknown') for r in self.reports)
        lines.append(f"**Detection Methods:**")
        for method, count in detection_methods.most_common():
            lines.append(f"- {method}: {count} queries")
        lines.append("")

        lines.append("---")
        lines.append("")

        # Section 3: Strategy Analysis
        lines.append("## 🎯 Strategy Analysis")
        lines.append("")

        strategy_counts = Counter(r.get('strategy', 'unknown') for r in self.reports)
        lines.append(f"**Strategy Usage:**")
        for strategy, count in strategy_counts.most_common():
            lines.append(f"- `{strategy}`: {count} queries")
        lines.append("")

        # Strategy effectiveness
        lines.append("**Strategy Effectiveness:**")
        strategy_results = defaultdict(list)
        for r in self.reports:
            strategy = r.get('strategy', 'unknown')
            chunks = r.get('chunks_retrieved', 0)
            strategy_results[strategy].append(chunks)

        for strategy, chunk_counts in strategy_results.items():
            avg = sum(chunk_counts) / len(chunk_counts) if chunk_counts else 0
            lines.append(f"- `{strategy}`: avg {avg:.1f} chunks/query")
        lines.append("")

        lines.append("---")
        lines.append("")

        # Section 4: Performance Analysis
        lines.append("## ⚡ Performance Analysis")
        lines.append("")

        # Timing analysis (if available)
        # For now, placeholder
        lines.append("**Response Times:**")
        lines.append("- Average retrieval time: [data needed from logs]")
        lines.append("- Average LLM time: [data needed from logs]")
        lines.append("")

        lines.append("---")
        lines.append("")

        # Section 5: Problem Areas
        lines.append("## ⚠️ Issues & Problem Areas")
        lines.append("")

        # Identify problematic queries
        problems = []

        for r in self.reports:
            query = r.get('query', 'Unknown')
            issues_found = []

            if r.get('chunks_retrieved', 0) == 0:
                issues_found.append("No chunks retrieved")

            if r.get('answer_length', 0) < 50:
                issues_found.append("Very short answer")

            if 'no information' in r.get('answer', '').lower():
                issues_found.append("LLM could not find information")

            if r.get('confidence', 1.0) < 0.5:
                issues_found.append(f"Low confidence ({r.get('confidence', 0):.2f})")

            if issues_found:
                problems.append((query, issues_found))

        if problems:
            lines.append(f"**{len(problems)} queries had issues:**")
            lines.append("")
            for query, issues in problems[:10]:  # Show top 10
                lines.append(f"**Query:** {query[:80]}")
                for issue in issues:
                    lines.append(f"- ⚠️ {issue}")
                lines.append("")
        else:
            lines.append("✅ No major issues detected!")
            lines.append("")

        lines.append("---")
        lines.append("")

        # Section 6: Successes
        lines.append("## ✅ Successful Queries")
        lines.append("")

        successes = [
            r for r in self.reports
            if r.get('chunks_retrieved', 0) > 0
            and r.get('answer_length', 0) >= 50
            and 'no information' not in r.get('answer', '').lower()
        ]

        lines.append(f"**{len(successes)} queries executed successfully ({len(successes)/len(self.reports)*100:.1f}%)**")
        lines.append("")

        if successes:
            lines.append("Sample successful queries:")
            for r in successes[:5]:
                lines.append(f"- {r.get('query', 'Unknown')[:80]}")
                lines.append(f"  - Intent: `{r.get('intent', 'N/A')}`, Chunks: {r.get('chunks_retrieved', 0)}, Answer: {r.get('answer_length', 0)} chars")
            lines.append("")

        lines.append("---")
        lines.append("")

        # Section 7: Recommendations
        lines.append("## 💡 Recommendations")
        lines.append("")

        recs = []

        if zero_results > len(self.reports) * 0.2:
            recs.append(f"**High zero-result rate ({zero_results} queries)** - Review intent detection and temporal filters")

        if no_info_answers > len(self.reports) * 0.3:
            recs.append(f"**Many 'no information' answers ({no_info_answers} queries)** - Retrieved chunks may not be relevant")

        if avg_confidence < 0.6:
            recs.append(f"**Low average intent confidence ({avg_confidence:.2f})** - Consider improving intent detection patterns")

        if avg_chunks < 3:
            recs.append(f"**Low average chunk count ({avg_chunks:.1f})** - May need to adjust retrieval parameters")

        if recs:
            for rec in recs:
                lines.append(f"- {rec}")
            lines.append("")
        else:
            lines.append("✅ No major issues identified - system performing well!")
            lines.append("")

        lines.append("---")
        lines.append("")

        # Section 8: Query List
        lines.append("## 📋 All Queries")
        lines.append("")
        lines.append("| # | Query | Intent | Chunks | Answer Len | Issues |")
        lines.append("|---|-------|--------|--------|------------|--------|")

        for i, r in enumerate(self.reports, 1):
            query = r.get('query', 'Unknown')[:40]
            intent = r.get('intent', 'N/A')
            chunks = r.get('chunks_retrieved', 0)
            ans_len = r.get('answer_length', 0)

            issues_flag = ""
            if chunks == 0 or ans_len < 50 or 'no information' in r.get('answer', '').lower():
                issues_flag = "⚠️"

            lines.append(f"| {i} | {query} | {intent} | {chunks} | {ans_len} | {issues_flag} |")

        lines.append("")

        return '\n'.join(lines)


def generate_reports(run_dir: Path, output_dir: Path) -> Dict:
    """
    Generate individual report for a single run.

    Args:
        run_dir: Path to run directory
        output_dir: Output directory for report

    Returns:
        Dictionary with report metadata
    """
    generator = QueryReportGenerator(run_dir)
    report_md = generator.generate_individual_report()

    # Save report
    output_dir.mkdir(parents=True, exist_ok=True)
    report_file = output_dir / f"{run_dir.name}_report.md"

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_md)

    # Extract key metadata for summary
    metadata = {
        'run_id': run_dir.name,
        'query': generator.metadata.get('query', 'Unknown'),
        'intent': generator.metadata.get('intent', {}).get('primary_intent', 'unknown'),
        'confidence': generator.metadata.get('intent', {}).get('confidence', 0),
        'detection_method': generator.metadata.get('intent', {}).get('detection_method', 'unknown'),
        'strategy': generator.metadata.get('strategy', 'unknown'),
        'chunks_retrieved': len(generator.chunks),
        'answer': generator.response or '',
        'answer_length': len(generator.response or ''),
        'report_file': str(report_file)
    }

    return metadata
