#!/usr/bin/env python3
"""
Debug script to find the actual project structure and locate BatchEmbedder.
"""

import os
import sys
from pathlib import Path

def find_files_by_name(root_dir: Path, filename: str, max_depth: int = 5):
    """Find all files with a specific name."""
    found_files = []
    
    def search_recursive(current_dir: Path, current_depth: int):
        if current_depth > max_depth:
            return
        
        try:
            for item in current_dir.iterdir():
                if item.is_file() and item.name == filename:
                    found_files.append(item)
                elif item.is_dir() and not item.name.startswith('.'):
                    search_recursive(item, current_depth + 1)
        except PermissionError:
            pass
    
    search_recursive(root_dir, 0)
    return found_files

def analyze_project_structure():
    """Analyze the project structure to find BatchEmbedder."""
    print("=" * 80)
    print("PROJECT STRUCTURE ANALYSIS")
    print("=" * 80)
    
    # Get current working directory
    cwd = Path.cwd()
    print(f"Current working directory: {cwd}")
    
    # Find batch_embedder.py files
    print("\nSearching for batch_embedder.py files...")
    batch_files = find_files_by_name(cwd, "batch_embedder.py")
    
    if batch_files:
        print(f"Found {len(batch_files)} batch_embedder.py files:")
        for i, file_path in enumerate(batch_files, 1):
            print(f"  {i}. {file_path}")
            
            # Try to determine the correct import path
            relative_path = file_path.relative_to(cwd)
            parts = relative_path.parts[:-1]  # Remove the .py filename
            
            # Convert path to import string
            if parts:
                import_path = ".".join(parts) + ".batch_embedder"
                print(f"     Possible import: from {import_path} import BatchEmbedder")
    else:
        print("❌ No batch_embedder.py files found!")
    
    # Search for any files containing "BatchEmbedder" class
    print("\nSearching for files containing 'BatchEmbedder' class...")
    python_files = []
    
    def find_python_files(directory: Path, max_depth: int = 5):
        def search_recursive(current_dir: Path, current_depth: int):
            if current_depth > max_depth:
                return
            
            try:
                for item in current_dir.iterdir():
                    if item.is_file() and item.suffix == '.py':
                        python_files.append(item)
                    elif item.is_dir() and not item.name.startswith('.'):
                        search_recursive(item, current_depth + 1)
            except PermissionError:
                pass
        
        search_recursive(directory, 0)
    
    find_python_files(cwd)
    
    batch_embedder_files = []
    for py_file in python_files:
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if 'class BatchEmbedder' in content:
                    batch_embedder_files.append(py_file)
        except:
            continue
    
    if batch_embedder_files:
        print(f"Found {len(batch_embedder_files)} files containing 'class BatchEmbedder':")
        for i, file_path in enumerate(batch_embedder_files, 1):
            print(f"  {i}. {file_path}")
    else:
        print("❌ No files containing 'class BatchEmbedder' found!")
    
    # Check Python path
    print(f"\nPython path (sys.path):")
    for i, path in enumerate(sys.path, 1):
        print(f"  {i}. {path}")
    
    # Look for scripts directory structure
    print("\nLooking for 'scripts' directory structure...")
    scripts_dirs = find_files_by_name(cwd, "scripts", max_depth=3)
    
    if scripts_dirs:
        scripts_dirs = [d for d in scripts_dirs if d.is_dir()]
        print(f"Found {len(scripts_dirs)} 'scripts' directories:")
        
        for scripts_dir in scripts_dirs:
            print(f"\n  📁 {scripts_dir}")
            
            # Check subdirectories
            try:
                for subdir in scripts_dir.iterdir():
                    if subdir.is_dir():
                        print(f"    📁 {subdir.name}/")
                        
                        # Check for api_clients or embeddings
                        if subdir.name in ['api_clients', 'embeddings']:
                            for sub_subdir in subdir.iterdir():
                                if sub_subdir.is_dir():
                                    print(f"      📁 {sub_subdir.name}/")
                                elif sub_subdir.is_file() and sub_subdir.suffix == '.py':
                                    print(f"      📄 {sub_subdir.name}")
            except PermissionError:
                print("    (Permission denied)")
    else:
        print("❌ No 'scripts' directories found!")

def test_possible_imports():
    """Test different possible import paths."""
    print("\n" + "=" * 80)
    print("TESTING POSSIBLE IMPORTS")
    print("=" * 80)
    
    # Common possible import paths
    possible_imports = [
        "scripts.api_clients.openai.batch_embedder",
        "api_clients.openai.batch_embedder", 
        "scripts.embeddings.batch_embedder",
        "embeddings.batch_embedder",
        "batch_embedder",
        "openai.batch_embedder",
        "scripts.api_clients.batch_embedder",
        "api_clients.batch_embedder"
    ]
    
    successful_imports = []
    
    for import_path in possible_imports:
        try:
            print(f"Trying: from {import_path} import BatchEmbedder")
            
            # Dynamic import
            module = __import__(import_path, fromlist=['BatchEmbedder'])
            BatchEmbedder = getattr(module, 'BatchEmbedder')
            
            print(f"  ✅ SUCCESS: {import_path}")
            successful_imports.append(import_path)
            
        except ImportError as e:
            print(f"  ❌ FAILED: {e}")
        except AttributeError as e:
            print(f"  ❌ NO CLASS: {e}")
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
    
    if successful_imports:
        print(f"\n🎉 SUCCESSFUL IMPORTS:")
        for imp in successful_imports:
            print(f"  from {imp} import BatchEmbedder")
    else:
        print("\n❌ NO SUCCESSFUL IMPORTS FOUND")

def check_current_directory_contents():
    """Check what's in the current directory."""
    print("\n" + "=" * 80)
    print("CURRENT DIRECTORY CONTENTS")
    print("=" * 80)
    
    cwd = Path.cwd()
    print(f"Contents of {cwd}:")
    
    try:
        items = sorted(cwd.iterdir())
        for item in items:
            if item.is_dir():
                print(f"  📁 {item.name}/")
            else:
                print(f"  📄 {item.name}")
    except PermissionError:
        print("  (Permission denied)")

def main():
    """Run all diagnostics."""
    print("BATCHEMBEDDER IMPORT DEBUG")
    print("=" * 80)
    
    check_current_directory_contents()
    analyze_project_structure()
    test_possible_imports()
    
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("1. Look at the successful imports above")
    print("2. Update your test script with the correct import path")
    print("3. Check if BatchEmbedder file exists where expected")
    print("4. Verify you're running from the correct directory")

if __name__ == "__main__":
    main()