"""
Email cleaning utilities.

Removes noise and duplicate content from emails before indexing.
"""

from .quote_deduplicator import QuoteDeduplicator
from .signature_detector import SignatureDetector

__all__ = ['QuoteDeduplicator', 'SignatureDetector']
