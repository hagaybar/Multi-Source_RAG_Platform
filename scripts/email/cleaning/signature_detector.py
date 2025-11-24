#!/usr/bin/env python3
"""
ML-Based Signature Detection for Emails

Detects and removes email signatures using pattern-based and ML-based approaches.

Problem:
    Email signatures add noise to vector embeddings and waste storage.
    Traditional regex patterns miss non-standard signatures.

Solution:
    1. Pattern-based detection (lightweight, fast)
    2. ML-based detection (future: fine-tuned DistilBERT)
    3. Structural analysis (common signature indicators)

Expected Impact:
    - 30-40% storage reduction (after quote dedup)
    - Better embedding quality (no signature noise)
    - Cleaner chunks for retrieval
"""

import re
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class SignatureMatch:
    """Represents a detected signature."""
    start_index: int
    end_index: int
    confidence: float  # 0-1
    method: str  # "pattern", "ml", "structural"
    signature_text: str


class SignatureDetector:
    """
    Detect and remove email signatures.

    Uses multiple strategies:
    1. Pattern-based: Common signature markers ("--", "Best regards", etc.)
    2. Structural: Position + formatting (last 5 lines, contact info, etc.)
    3. ML-based: (Future) Fine-tuned classifier for signature vs content
    """

    def __init__(
        self,
        min_signature_length: int = 10,
        max_signature_length: int = 500,
        use_ml: bool = False,
        confidence_threshold: float = 0.7
    ):
        """
        Initialize signature detector.

        Args:
            min_signature_length: Minimum characters for signature
            max_signature_length: Maximum characters for signature
            use_ml: Enable ML-based detection (requires model)
            confidence_threshold: Minimum confidence to accept detection
        """
        self.min_length = min_signature_length
        self.max_length = max_signature_length
        self.use_ml = use_ml
        self.confidence_threshold = confidence_threshold

        # Common signature delimiters
        self.signature_delimiters = [
            r'^--\s*$',  # Standard delimiter
            r'^-- $',    # Standard with space
            r'^_{20,}$', # Underscore lines
            r'^={20,}$', # Equal lines
            r'^-{20,}$', # Dash lines
        ]

        # Signature start patterns (case-insensitive)
        self.signature_patterns = [
            r'(?i)^(best|kind|warm|regards|sincerely|thanks|cheers|cordially)',
            r'(?i)^sent from my (iphone|ipad|android|mobile)',
            r'(?i)^get outlook for (ios|android)',
            r'(?i)^this email and any files',  # Legal disclaimers
            r'(?i)^confidential communication',
            r'(?i)^please consider the environment',
            r'(?i)^this message (contains|is intended)',
        ]

        # Contact info patterns (phone, email, URL)
        self.contact_patterns = [
            r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',  # Phone
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'https?://[^\s]+',  # URL
            r'(?i)(tel|phone|fax|mobile|cell):?\s*[\d\s\(\)\-\.+]+',  # Phone labels
        ]

        # Legal/disclaimer keywords
        self.disclaimer_keywords = [
            'confidential', 'privileged', 'intended recipient',
            'unauthorized', 'disclosure', 'confidentiality',
            'legal', 'attorney', 'copyright', 'disclaimer',
            'virus', 'malware', 'liability'
        ]

    def detect_signature(
        self,
        text: str,
        aggressive: bool = False
    ) -> Tuple[str, Optional[str]]:
        """
        Split text into (content, signature).

        Args:
            text: Email body text
            aggressive: Use more aggressive detection (may remove content)

        Returns:
            (content_without_signature, signature_text)
            If no signature detected, signature_text is None
        """
        if not text or len(text) < self.min_length:
            return text, None

        # Try detection methods in order of confidence
        matches: List[SignatureMatch] = []

        # 1. Delimiter-based detection (highest confidence)
        delimiter_match = self._detect_by_delimiter(text)
        if delimiter_match:
            matches.append(delimiter_match)

        # 2. Pattern-based detection
        pattern_match = self._detect_by_pattern(text)
        if pattern_match:
            matches.append(pattern_match)

        # 3. Structural detection (position + features)
        structural_match = self._detect_by_structure(text, aggressive)
        if structural_match:
            matches.append(structural_match)

        # 4. ML-based detection (future)
        if self.use_ml:
            ml_match = self._detect_by_ml(text)
            if ml_match:
                matches.append(ml_match)

        # Select best match above threshold
        best_match = self._select_best_match(matches)

        if best_match:
            content = text[:best_match.start_index].rstrip()
            signature = best_match.signature_text
            return content, signature
        else:
            return text, None

    def _detect_by_delimiter(self, text: str) -> Optional[SignatureMatch]:
        """Detect signature by standard delimiters (-- )."""
        lines = text.split('\n')

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Check for delimiter patterns
            for pattern in self.signature_delimiters:
                if re.match(pattern, stripped):
                    # Found delimiter - everything after is signature
                    signature_lines = lines[i:]
                    signature_text = '\n'.join(signature_lines)

                    # Validate length
                    if self.min_length <= len(signature_text) <= self.max_length:
                        start_index = sum(len(l) + 1 for l in lines[:i])
                        return SignatureMatch(
                            start_index=start_index,
                            end_index=len(text),
                            confidence=0.95,
                            method="delimiter",
                            signature_text=signature_text
                        )

        return None

    def _detect_by_pattern(self, text: str) -> Optional[SignatureMatch]:
        """Detect signature by common closing patterns."""
        lines = text.split('\n')
        total_lines = len(lines)

        # Only check last 30% of email (signatures at end)
        start_search = max(0, int(total_lines * 0.7))

        for i in range(start_search, total_lines):
            line = lines[i].strip()

            # Check signature patterns
            for pattern in self.signature_patterns:
                if re.match(pattern, line):
                    # Found signature start
                    signature_lines = lines[i:]
                    signature_text = '\n'.join(signature_lines)

                    # Validate length
                    if self.min_length <= len(signature_text) <= self.max_length:
                        start_index = sum(len(l) + 1 for l in lines[:i])
                        return SignatureMatch(
                            start_index=start_index,
                            end_index=len(text),
                            confidence=0.85,
                            method="pattern",
                            signature_text=signature_text
                        )

        return None

    def _detect_by_structure(
        self,
        text: str,
        aggressive: bool = False
    ) -> Optional[SignatureMatch]:
        """
        Detect signature by structural features.

        Features:
        - Position (last 5-15 lines)
        - Contact info density (emails, phones, URLs)
        - Disclaimer keywords
        - Short lines (< 60 chars)
        """
        lines = text.split('\n')
        total_lines = len(lines)

        if total_lines < 5:
            return None  # Too short

        # Analyze last 15 lines
        window_size = min(15, total_lines)
        start_search = max(0, total_lines - window_size)

        for i in range(start_search, total_lines - 2):  # At least 3 lines
            candidate_lines = lines[i:]
            candidate_text = '\n'.join(candidate_lines)

            if not (self.min_length <= len(candidate_text) <= self.max_length):
                continue

            # Calculate feature scores
            contact_score = self._score_contact_info(candidate_text)
            disclaimer_score = self._score_disclaimer(candidate_text)
            length_score = self._score_line_lengths(candidate_lines)

            # Weighted scoring
            total_score = (
                contact_score * 0.4 +
                disclaimer_score * 0.3 +
                length_score * 0.3
            )

            # Threshold depends on aggressive mode
            threshold = 0.5 if aggressive else 0.65

            if total_score >= threshold:
                start_index = sum(len(l) + 1 for l in lines[:i])
                return SignatureMatch(
                    start_index=start_index,
                    end_index=len(text),
                    confidence=total_score,
                    method="structural",
                    signature_text=candidate_text
                )

        return None

    def _score_contact_info(self, text: str) -> float:
        """Score based on contact info density."""
        matches = 0
        for pattern in self.contact_patterns:
            matches += len(re.findall(pattern, text))

        # Normalize by text length
        density = matches / (len(text) / 100)  # Matches per 100 chars
        return min(1.0, density / 2)  # Cap at 1.0

    def _score_disclaimer(self, text: str) -> float:
        """Score based on legal disclaimer keywords."""
        text_lower = text.lower()
        keyword_count = sum(
            1 for keyword in self.disclaimer_keywords
            if keyword in text_lower
        )

        # Normalize
        return min(1.0, keyword_count / 3)  # 3+ keywords = 1.0

    def _score_line_lengths(self, lines: List[str]) -> float:
        """Score based on line length distribution (signatures have short lines)."""
        if not lines:
            return 0.0

        # Count lines under 60 chars
        short_lines = sum(1 for line in lines if len(line.strip()) < 60)
        ratio = short_lines / len(lines)

        # Signatures typically 70%+ short lines
        if ratio >= 0.7:
            return 1.0
        elif ratio >= 0.5:
            return 0.6
        else:
            return 0.3

    def _detect_by_ml(self, text: str) -> Optional[SignatureMatch]:
        """
        ML-based signature detection (future implementation).

        Would use fine-tuned DistilBERT classifier trained on:
        - Enron email dataset (labeled signatures)
        - Custom labeled email corpus

        Features:
        - Sentence embeddings
        - Position encoding
        - Structural features
        """
        # Placeholder for future ML implementation
        # For now, return None
        return None

    def _select_best_match(
        self,
        matches: List[SignatureMatch]
    ) -> Optional[SignatureMatch]:
        """Select best signature match based on confidence."""
        if not matches:
            return None

        # Filter by confidence threshold
        valid_matches = [
            m for m in matches
            if m.confidence >= self.confidence_threshold
        ]

        if not valid_matches:
            return None

        # Prefer delimiter > pattern > structural > ml
        method_priority = {
            "delimiter": 4,
            "pattern": 3,
            "structural": 2,
            "ml": 1
        }

        # Sort by priority, then confidence
        valid_matches.sort(
            key=lambda m: (method_priority.get(m.method, 0), m.confidence),
            reverse=True
        )

        return valid_matches[0]

    def get_stats(self, original: str, cleaned: str, signature: Optional[str]) -> dict:
        """Calculate signature removal statistics."""
        removed_chars = len(signature) if signature else 0
        removed_lines = signature.count('\n') + 1 if signature else 0

        return {
            'original_length': len(original),
            'cleaned_length': len(cleaned),
            'removed_chars': removed_chars,
            'removed_lines': removed_lines,
            'reduction_ratio': removed_chars / len(original) if len(original) > 0 else 0,
            'had_signature': signature is not None
        }


def test_signature_detector():
    """Test signature detector with examples."""

    # Example 1: Standard delimiter
    email1 = """
Hi team,

Thanks for the update. I'll review the proposal tomorrow.

--
John Smith
Senior Manager
Company Inc.
john.smith@company.com
+1 (555) 123-4567
"""

    # Example 2: "Best regards" pattern
    email2 = """
The project is on track for Q4 delivery.

Best regards,
Alice Johnson
alice@example.com
"""

    # Example 3: Legal disclaimer
    email3 = """
Please find the report attached.

This email and any files transmitted with it are confidential and
intended solely for the use of the individual or entity to whom they
are addressed. If you have received this email in error please notify
the system manager. This message contains confidential information.
"""

    detector = SignatureDetector()

    print("=" * 80)
    print("SIGNATURE DETECTION TESTS")
    print("=" * 80)

    for i, email in enumerate([email1, email2, email3], 1):
        print(f"\n{'─' * 80}")
        print(f"Test {i}")
        print(f"{'─' * 80}\n")

        content, signature = detector.detect_signature(email.strip())
        stats = detector.get_stats(email.strip(), content, signature)

        print(f"Original length: {stats['original_length']} chars")
        print(f"Signature detected: {stats['had_signature']}")
        print(f"Removed: {stats['removed_chars']} chars ({stats['reduction_ratio']:.1%})")
        print(f"Removed lines: {stats['removed_lines']}")

        print(f"\nContent (cleaned):")
        print(f"  {repr(content[:100])}")

        if signature:
            print(f"\nSignature (removed):")
            print(f"  {repr(signature[:100])}")


if __name__ == "__main__":
    test_signature_detector()
