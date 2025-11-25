import json

# Check for duplicates and size distribution
chunks_seen = {}
token_sizes = []

with open("data/projects/Primo_List/input/chunks_outlook_eml.tsv", 'r') as f:
    lines = f.readlines()[1:]  # Skip header
    
    for i, line in enumerate(lines[:100], 1):  # Check first 100
        parts = line.strip().split('\t')
        if len(parts) >= 4:
            chunk_id, doc_id, text, tokens = parts[:4]
            
            # Track text duplicates
            if text in chunks_seen:
                chunks_seen[text] += 1
            else:
                chunks_seen[text] = 1
            
            # Track sizes
            try:
                token_sizes.append(int(tokens))
            except:
                pass

print(f"First 100 chunks analyzed:")
print(f"  Unique texts: {len(chunks_seen)}")
print(f"  Total chunks: {len(token_sizes)}")
print(f"  Duplicates: {sum(1 for c in chunks_seen.values() if c > 1)}")
print(f"  Avg tokens: {sum(token_sizes)/len(token_sizes):.1f}")
print(f"  Token range: {min(token_sizes)}-{max(token_sizes)}")

if len(chunks_seen) < len(token_sizes) * 0.5:
    print("\n⚠️  HIGH DUPLICATION DETECTED!")
