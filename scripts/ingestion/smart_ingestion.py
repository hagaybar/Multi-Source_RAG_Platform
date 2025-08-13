# scripts/ingestion/smart_ingestion.py
"""
This script serves as a testbed for the dynamic chunking system. It uses the
document_classifier.py to automatically determine the chunking strategy for
different types of documents and then calls the refactored chunking logic.
The output is enhanced to provide detailed information for verification.
"""

import pprint
from scripts.chunking.document_classifier import classify_document
from scripts.chunking.chunker_v3 import split

# --- Test Cases ---

# Test Case 1: A "how-to" guide
how_to_text = """
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

# Test Case 2: A presentation
presentation_text = """
Q2 Financial Results

Agenda
- Welcome and Introduction
- Q2 Financial Highlights
- Live Demo of New Product
- Q&A
"""

# Test Case 3: A manual
manual_text = """
User Manual: Super Widget

Table of Contents
1. Introduction
2. Installation
3. Operation
4. Troubleshooting
"""

# Test Case 4: A generic document (fallback)
generic_text = """
A random document about nothing in particular.
There are no special keywords here.
Just some plain text to see what happens.
"""

test_cases = [
    {
        "name": "How-To Guide Test",
        "text": how_to_text,
        "expected_doc_type": "how-to",
    },
    {
        "name": "Presentation Test",
        "text": presentation_text,
        "expected_doc_type": "presentation",
    },
    {
        "name": "Manual Test",
        "text": manual_text,
        "expected_doc_type": "manual",
    },
    {
        "name": "Generic Document Fallback Test",
        "text": generic_text,
        "expected_doc_type": "general",
    },
]

def main():
    """
    Main function to run the smart ingestion process for all test cases.
    """
    print("--- Starting Smart Ingestion Testbed ---")

    for i, test_case in enumerate(test_cases):
        print(f"\n{'='*50}")
        print(f"--- Running Test Case {i+1}: {test_case['name']} ---")
        print(f"--- Expected doc_type: {test_case['expected_doc_type']} ---")
        print(f"{'='*50}\n")

        text = test_case["text"]

        # 1. Classify the document
        classification = classify_document(text)
        doc_type = classification['doc_type']
        strategy = classification['strategy']

        print(f"Recommended doc_type: {doc_type}")
        print(f"Recommended strategy: {strategy}")

        # 2. Prepare metadata
        meta = {
            "doc_id": f"test-doc-{i+1}",
            "doc_type": doc_type,
            "source": "smart_ingestion_testbed"
        }

        # 3. Chunk the document
        chunks = split(text, meta)

        # 4. Print the results
        print(f"\n--- Found {len(chunks)} chunks ---")
        for chunk in chunks:
            print("\n--- Chunk ---")
            print(f"  ID: {chunk.id}")
            # Print the first 50 characters of the text
            print(f"  Text: '{chunk.text[:50].strip()}...'")
            if hasattr(chunk, 'parent_id') and chunk.parent_id:
                print(f"  Parent ID: {chunk.parent_id}")
            else:
                print("  Parent ID: None")


if __name__ == "__main__":
    main()
