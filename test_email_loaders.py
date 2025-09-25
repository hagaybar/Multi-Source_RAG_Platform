#!/usr/bin/env python3
"""
Quick test script for new email loaders.
Tests the import and basic functionality without requiring actual email files.
"""

import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

def test_imports():
    """Test that all email loaders can be imported."""
    print("Testing email loader imports...")
    
    try:
        from scripts.ingestion.email_loader import load_eml, load_msg, load_mbox, HAS_MSG_SUPPORT, HAS_PST_SUPPORT
        print("✅ All email loaders imported successfully")
        print(f"📧 MSG support available: {HAS_MSG_SUPPORT}")
        print(f"📁 PST support available: {HAS_PST_SUPPORT}")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_registry():
    """Test that email formats are registered."""
    print("\nTesting loader registry...")
    
    try:
        from scripts.ingestion import LOADER_REGISTRY
        
        email_formats = [".eml", ".msg", ".mbox"]
        missing_formats = []
        
        for fmt in email_formats:
            if fmt in LOADER_REGISTRY:
                print(f"✅ {fmt} format registered")
            else:
                missing_formats.append(fmt)
                print(f"❌ {fmt} format NOT registered")
        
        if not missing_formats:
            print("✅ All email formats properly registered")
            return True
        else:
            print(f"❌ Missing formats: {missing_formats}")
            return False
            
    except Exception as e:
        print(f"❌ Registry test failed: {e}")
        return False

def test_chunking_rules():
    """Test that chunking rules exist for email formats."""
    print("\nTesting chunking rules...")
    
    try:
        import yaml
        
        config_path = ROOT / "configs" / "chunk_rules.yaml"
        with open(config_path, 'r') as f:
            rules = yaml.safe_load(f)
        
        email_formats = ["eml", "msg", "mbox"]
        missing_rules = []
        
        for fmt in email_formats:
            if fmt in rules:
                strategy = rules[fmt].get('strategy', 'unknown')
                print(f"✅ {fmt}: {strategy}")
            else:
                missing_rules.append(fmt)
                print(f"❌ {fmt}: No chunking rule found")
        
        if not missing_rules:
            print("✅ All email formats have chunking rules")
            return True
        else:
            print(f"❌ Missing rules for: {missing_rules}")
            return False
            
    except Exception as e:
        print(f"❌ Chunking rules test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Email Integration Features\n")
    
    results = []
    results.append(test_imports())
    results.append(test_registry())
    results.append(test_chunking_rules())
    
    print(f"\n📊 Test Results: {sum(results)}/{len(results)} passed")
    
    if all(results):
        print("\n🎉 All tests passed! Email integration is ready.")
        print("\n📋 Next steps:")
        print("1. Create sample email files (.msg and .mbox)")
        print("2. Test actual file processing through the UI")
        print("3. Verify chunking and embedding works correctly")
        return True
    else:
        print("\n⚠️  Some tests failed. Please review the output above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)