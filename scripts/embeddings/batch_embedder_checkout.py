#!/usr/bin/env python3
"""
Direct test of BatchEmbedder to isolate any issues.
Run this independently to test if BatchEmbedder can reach OpenAI.
"""

import os
import sys
from pathlib import Path

# CRITICAL: Add the project root to Python path
project_root = Path(__file__).parent  # This should be the Multi-Source_RAG_Platform directory
print(f"DEBUG: Adding to Python path: {project_root}")
sys.path.insert(0, str(project_root))

# Verify the path was added
print(f"DEBUG: Python path now includes: {project_root in [Path(p) for p in sys.path]}")
print(f"DEBUG: First few sys.path entries:")
for i, path in enumerate(sys.path[:5]):
    print(f"  {i+1}. {path}")

def test_environment():
    """Test environment variables and API key."""
    print("\n" + "=" * 80)
    print("ENVIRONMENT TEST")
    print("=" * 80)
    
    openai_key = os.getenv("OPEN_AI")
    openai_key_alt = os.getenv("OPENAI_API_KEY")
    
    print(f"OPEN_AI env var: {bool(openai_key)}")
    print(f"OPENAI_API_KEY env var: {bool(openai_key_alt)}")
    
    if openai_key:
        print(f"OPEN_AI starts with: {openai_key[:10]}...")
    if openai_key_alt:
        print(f"OPENAI_API_KEY starts with: {openai_key_alt[:10]}...")
    
    if not openai_key and not openai_key_alt:
        print("ERROR: No OpenAI API key found in environment!")
        return False
    
    return True

def test_import():
    """Test importing BatchEmbedder."""
    print("\n" + "=" * 80)
    print("IMPORT TEST")
    print("=" * 80)
    
    try:
        print("Attempting: from scripts.api_clients.openai.batch_embedder import BatchEmbedder")
        from scripts.api_clients.openai.batch_embedder import BatchEmbedder
        print("✓ BatchEmbedder import successful!")
        print(f"✓ BatchEmbedder class: {BatchEmbedder}")
        return BatchEmbedder
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        
        # Try alternative approach - direct file import
        print("\nTrying alternative approach...")
        batch_embedder_path = project_root / "scripts" / "api_clients" / "openai" / "batch_embedder.py"
        print(f"Looking for file at: {batch_embedder_path}")
        print(f"File exists: {batch_embedder_path.exists()}")
        
        if batch_embedder_path.exists():
            # Import using importlib
            import importlib.util
            spec = importlib.util.spec_from_file_location("batch_embedder", batch_embedder_path)
            batch_embedder_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(batch_embedder_module)
            BatchEmbedder = batch_embedder_module.BatchEmbedder
            print("✓ Alternative import successful!")
            return BatchEmbedder
        
        return None
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return None

def test_openai_direct():
    """Test OpenAI client directly."""
    print("\n" + "=" * 80)
    print("DIRECT OPENAI CLIENT TEST")
    print("=" * 80)
    
    try:
        from openai import OpenAI
        print("✓ OpenAI import successful")
        
        api_key = os.getenv("OPEN_AI") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("✗ No API key available for testing")
            return False
            
        client = OpenAI(api_key=api_key)
        print("✓ OpenAI client created")
        
        # Test a simple API call
        print("Testing simple embedding call...")
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=["Hello world test"]
        )
        print(f"✓ Direct embedding call successful: {len(response.data[0].embedding)} dimensions")
        return True
        
    except Exception as e:
        print(f"✗ Direct OpenAI test failed: {e}")
        return False

def test_batch_embedder(BatchEmbedder):
    """Test BatchEmbedder directly."""
    print("\n" + "=" * 80)
    print("BATCHEMBEDDER TEST")
    print("=" * 80)
    
    if BatchEmbedder is None:
        print("✗ BatchEmbedder class not available")
        return False
    
    try:
        # Create test directory
        test_dir = Path("./debug_test_output")
        test_dir.mkdir(exist_ok=True)
        print(f"✓ Test directory created: {test_dir}")
        
        # Create BatchEmbedder instance
        batch_embedder = BatchEmbedder(
            model="text-embedding-3-large",
            output_dir=test_dir
        )
        print("✓ BatchEmbedder instance created")
        
        # Test with small batch
        test_texts = [
            "This is a test embedding",
            "Another test sentence",
            "Final test text"
        ]
        test_ids = ["test-1", "test-2", "test-3"]
        
        print(f"Testing with {len(test_texts)} texts...")
        print("*** IF THIS HANGS, CHECK YOUR OPENAI DASHBOARD ***")
        print("*** This will make real API calls and may take time ***")
        
        result = batch_embedder.run(test_texts, test_ids)
        
        print(f"✓ BatchEmbedder test successful!")
        print(f"✓ Returned {len(result)} embeddings")
        
        # Verify results
        for test_id in test_ids:
            if test_id in result:
                embedding = result[test_id]
                print(f"✓ ID {test_id}: {len(embedding)} dimensions")
            else:
                print(f"✗ Missing embedding for ID: {test_id}")
        
        return True
        
    except Exception as e:
        print(f"✗ BatchEmbedder test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("BATCH EMBEDDER DEBUG TEST")
    print("=" * 80)
    print(f"Running from: {Path.cwd()}")
    print(f"Project root: {project_root}")
    
    success = True
    
    # Test 1: Import
    BatchEmbedder = test_import()
    if BatchEmbedder is None:
        success = False
    
    # Test 2: Environment
    if not test_environment():
        success = False
    
    # Test 3: Direct OpenAI (optional, costs money)
    print("\nWARNING: The next test will make a real OpenAI API call and cost money.")
    response = input("Do you want to test direct OpenAI API? (y/N): ").strip().lower()
    if response == 'y':
        if not test_openai_direct():
            success = False
    else:
        print("Skipped direct OpenAI test")
    
    # Test 4: BatchEmbedder (also costs money)
    if BatchEmbedder:
        print("\nWARNING: The next test will make real OpenAI batch API calls and cost money.")
        response = input("Do you want to test BatchEmbedder? (y/N): ").strip().lower()
        if response == 'y':
            if not test_batch_embedder(BatchEmbedder):
                success = False
        else:
            print("Skipped BatchEmbedder test")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if success:
        print("✓ CRITICAL TESTS PASSED")
        print("✓ BatchEmbedder can be imported and should work")
    else:
        print("✗ SOME TESTS FAILED")
        print("✗ Fix the failing tests before running the main application")
    
    if BatchEmbedder:
        print(f"\n📝 CORRECT IMPORT PATH FOUND:")
        print(f"   from scripts.api_clients.openai.batch_embedder import BatchEmbedder")

if __name__ == "__main__":
    main()