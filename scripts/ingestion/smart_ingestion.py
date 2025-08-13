# scripts/ingestion/smart_ingestion.py
"""
This script demonstrates the new smart ingestion pipeline. It uses the
document classifier to determine the chunking strategy and then calls the
refactored chunking logic.
"""

import pprint
from scripts.chunking.document_classifier import classify_document
from scripts.chunking.chunker_v3 import split

# Sample document text - a "how-to" guide
sample_document_text = """
My How-To Guide

This guide will walk you through the process of setting up your new device.

Step 1: Unbox the device.
Carefully open the box and remove the device and all accessories.

Step 2: Connect the power cable.
Plug the power cable into the device and then into a wall outlet.

Step 3: Turn on the device.
Press the power button to turn on the device. The screen should light up.

---
Additional Notes:
- Keep the device away from water.
- Do not expose to extreme temperatures.
"""

def main():
    """
    Main function to run the smart ingestion process.
    """
    print("--- Starting Smart Ingestion ---")

    # 1. Classify the document
    classification = classify_document(sample_document_text)
    doc_type = classification['doc_type']

    print(f"Document classified as: {doc_type}")

    # 2. Prepare metadata
    meta = {
        "doc_id": "sample-doc-123",
        "doc_type": doc_type,
        "source": "smart_ingestion_test"
    }

    # 3. Chunk the document
    chunks = split(sample_document_text, meta)

    # 4. Print the results
    print(f"\n--- Found {len(chunks)} chunks ---")
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i+1} ---")
        pprint.pprint(chunk)

if __name__ == "__main__":
    main()
