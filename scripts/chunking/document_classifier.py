# scripts/chunking/document_classifier.py
"""
This module contains a simple document classifier that uses rule-based logic
to determine the document type and a recommended chunking strategy.
"""

from typing import Dict

def classify_document(text: str) -> Dict[str, str]:
    """
    Classifies a document based on simple keyword analysis.

    Args:
        text: The text of the document.

    Returns:
        A dictionary containing the recommended 'doc_type' and 'strategy'.
    """
    text_lower = text.lower()

    if "agenda" in text_lower and "live demo" in text_lower:
        return {"doc_type": "presentation", "strategy": "by_slide"}

    if "table of contents" in text_lower:
        return {"doc_type": "manual", "strategy": "by_paragraph"}

    if "step 1" in text_lower and "step 2" in text_lower:
        return {"doc_type": "how-to", "strategy": "by_paragraph"}

    # Default fallback
    return {"doc_type": "general", "strategy": "by_paragraph"}
