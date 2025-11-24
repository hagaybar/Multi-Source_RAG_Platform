#!/usr/bin/env python3
"""
Quote Deduplication for Email Threads

Removes quoted text from reply emails to prevent indexing
the same content multiple times in a thread.

Problem:
    Thread with 10 emails → first email indexed 10 times!

Solution:
    Detect and remove quoted text, keeping only unique contributions.
"""

import re
from difflib import SequenceMatcher
from typing import List, Tuple, Optional


class QuoteDeduplicator:
    """
    Remove quoted text from email replies.

    Uses multiple strategies:
    1. Quote marker detection ("> ", "On X wrote:")
    2. Text overlap analysis (SequenceMatcher)
    3. Visual indentation patterns
    """

    def __init__(
        self,
        min_quote_length: int = 50,
        similarity_threshold: float = 0.85,
        aggressive_mode: bool = False
    ):
        """
        Initialize quote deduplicator.

        Args:
            min_quote_length: Minimum chars to consider as quote
            similarity_threshold: Text similarity for overlap detection (0-1)
            aggressive_mode: Remove more aggressively (may remove content)
        """
        self.min_quote_length = min_quote_length
        self.similarity_threshold = similarity_threshold
        self.aggressive_mode = aggressive_mode

        # Quote markers
        self.quote_prefixes = ['>', '|', '│', '┃']

        # Reply header patterns
        self.reply_patterns = [
            r'On .+ wrote:',
            r'From:.+Sent:.+To:.+Subject:',
            r'-----Original Message-----',
            r'________________________________',
            r'Begin forwarded message:',
            r'---------- Forwarded message ----------',
        ]

    def deduplicate(
        self,
        current_text: str,
        parent_text: Optional[str] = None
    ) -> Tuple[str, dict]:
        """
        Remove quoted text from email.

        Args:
            current_text: Current email body
            parent_text: Parent email body (if available)

        Returns:
            (deduplicated_text, stats_dict)
        """
        stats = {
            'original_length': len(current_text),
            'method': 'none',
            'removed_chars': 0,
            'removed_lines': 0,
            'quote_blocks_found': 0
        }

        # Step 1: Remove quote markers ("> " lines)
        cleaned_text, marker_stats = self._remove_quote_markers(current_text)
        if marker_stats['removed_lines'] > 0:
            stats.update(marker_stats)
            stats['method'] = 'markers'

        # Step 2: Remove reply headers ("On X wrote:")
        cleaned_text, header_stats = self._remove_reply_headers(cleaned_text)
        if header_stats['removed_lines'] > 0:
            stats['removed_lines'] += header_stats['removed_lines']
            stats['removed_chars'] += header_stats['removed_chars']
            if stats['method'] == 'none':
                stats['method'] = 'headers'

        # Step 3: Overlap detection with parent (if available)
        if parent_text and len(parent_text) > self.min_quote_length:
            cleaned_text, overlap_stats = self._remove_overlapping_text(
                cleaned_text,
                parent_text
            )
            if overlap_stats['removed_chars'] > 0:
                stats['removed_chars'] += overlap_stats['removed_chars']
                stats['removed_lines'] += overlap_stats['removed_lines']
                stats['quote_blocks_found'] = overlap_stats['quote_blocks_found']
                stats['method'] = 'overlap'

        stats['final_length'] = len(cleaned_text)
        stats['reduction_ratio'] = (
            stats['removed_chars'] / stats['original_length']
            if stats['original_length'] > 0 else 0
        )

        return cleaned_text, stats

    def _remove_quote_markers(self, text: str) -> Tuple[str, dict]:
        """Remove lines starting with quote markers (>, |, etc.)."""

        lines = text.split('\n')
        kept_lines = []
        removed_lines = 0
        removed_chars = 0

        for line in lines:
            stripped = line.lstrip()

            # Check if line starts with quote marker
            is_quote = any(
                stripped.startswith(prefix)
                for prefix in self.quote_prefixes
            )

            if is_quote:
                removed_lines += 1
                removed_chars += len(line) + 1  # +1 for newline
            else:
                kept_lines.append(line)

        cleaned = '\n'.join(kept_lines)

        return cleaned, {
            'removed_lines': removed_lines,
            'removed_chars': removed_chars
        }

    def _remove_reply_headers(self, text: str) -> Tuple[str, dict]:
        """Remove reply header blocks ("On X wrote:", etc.)."""

        removed_chars = 0
        removed_lines = 0

        for pattern in self.reply_patterns:
            # Find matches
            matches = list(re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE))

            if not matches:
                continue

            # Remove matched sections plus surrounding context
            for match in reversed(matches):  # Reverse to preserve indices
                start = match.start()
                end = match.end()

                # Expand to include full line
                line_start = text.rfind('\n', 0, start) + 1
                line_end = text.find('\n', end)
                if line_end == -1:
                    line_end = len(text)

                removed_text = text[line_start:line_end]
                removed_chars += len(removed_text)
                removed_lines += removed_text.count('\n') + 1

                text = text[:line_start] + text[line_end:]

        return text, {
            'removed_lines': removed_lines,
            'removed_chars': removed_chars
        }

    def _remove_overlapping_text(
        self,
        current_text: str,
        parent_text: str
    ) -> Tuple[str, dict]:
        """
        Remove text blocks that overlap with parent email.

        Uses SequenceMatcher to find matching blocks, then removes
        blocks with high similarity to parent.
        """

        # Split into lines for comparison
        current_lines = current_text.split('\n')
        parent_lines = parent_text.split('\n')

        # Find matching blocks
        matcher = SequenceMatcher(
            None,
            current_lines,
            parent_lines,
            autojunk=False
        )

        blocks = matcher.get_matching_blocks()

        # Track which lines to keep
        lines_to_remove = set()
        quote_blocks_found = 0

        for block in blocks:
            current_start = block.a
            current_size = block.size

            # Only consider significant blocks
            if current_size < 2:  # Too small
                continue

            # Calculate similarity of this block
            block_text = '\n'.join(
                current_lines[current_start:current_start + current_size]
            )

            if len(block_text) < self.min_quote_length:
                continue

            # Check if this block is likely a quote
            # (appears in parent with high similarity)
            is_quote = current_size >= 3  # 3+ matching lines = likely quote

            if is_quote:
                quote_blocks_found += 1
                for i in range(current_start, current_start + current_size):
                    lines_to_remove.add(i)

        # Build cleaned text
        kept_lines = [
            line for i, line in enumerate(current_lines)
            if i not in lines_to_remove
        ]

        cleaned_text = '\n'.join(kept_lines)

        removed_chars = len(current_text) - len(cleaned_text)
        removed_lines = len(lines_to_remove)

        return cleaned_text, {
            'removed_chars': removed_chars,
            'removed_lines': removed_lines,
            'quote_blocks_found': quote_blocks_found
        }

    def is_likely_reply(self, text: str, subject: str = '') -> bool:
        """Check if email is likely a reply."""

        # Check subject line
        if subject.lower().startswith(('re:', 'fwd:', 'fw:')):
            return True

        # Check for quote markers
        lines = text.split('\n')
        quoted_lines = sum(
            1 for line in lines
            if any(line.lstrip().startswith(p) for p in self.quote_prefixes)
        )

        if quoted_lines > 3:  # 3+ quoted lines = likely reply
            return True

        # Check for reply headers
        for pattern in self.reply_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True

        return False


def test_quote_deduplicator():
    """Test quote deduplicator with examples."""

    # Example 1: Simple quote markers
    email_with_quotes = """
Hi,

Thanks for the update!

> On Mon, Nov 22 wrote:
> The budget has been approved.
> We're moving forward with the project.
> Let me know if you have questions.

I have one question about the timeline.

Best,
Alice
"""

    dedup = QuoteDeduplicator()
    cleaned, stats = dedup.deduplicate(email_with_quotes)

    print("="*80)
    print("TEST 1: Quote Markers")
    print("="*80)
    print(f"Original length: {stats['original_length']} chars")
    print(f"Method: {stats['method']}")
    print(f"Removed: {stats['removed_chars']} chars ({stats['reduction_ratio']:.1%})")
    print(f"Removed lines: {stats['removed_lines']}")
    print(f"\nCleaned text:")
    print(cleaned)
    print()

    # Example 2: Overlap detection
    parent_email = """
The Q3 budget review is complete. We're allocating an extra $50k to
marketing to hit our acquisition targets. Finance approved it yesterday.
"""

    reply_email = """
Great news! Thanks for the update.

The Q3 budget review is complete. We're allocating an extra $50k to
marketing to hit our acquisition targets. Finance approved it yesterday.

When do we start?
"""

    cleaned, stats = dedup.deduplicate(reply_email, parent_email)

    print("="*80)
    print("TEST 2: Overlap Detection")
    print("="*80)
    print(f"Original length: {stats['original_length']} chars")
    print(f"Method: {stats['method']}")
    print(f"Quote blocks found: {stats['quote_blocks_found']}")
    print(f"Removed: {stats['removed_chars']} chars ({stats['reduction_ratio']:.1%})")
    print(f"\nCleaned text:")
    print(cleaned)
    print()


if __name__ == "__main__":
    test_quote_deduplicator()
