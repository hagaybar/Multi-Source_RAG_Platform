from pathlib import Path
import json, csv
from scripts.core.project_manager import ProjectManager
from scripts.chunking.models import Chunk
from scripts.agents.image_insight_agent import ImageInsightAgent

# === Configure test ===
# /home/hagaybar/projects/Multi-Source_RAG_Platform/data/projects/demo-image-ingest/input/chunks_pptx.tsv
project_dir = Path("data/projects/demo-image-ingest")  # Change to your test project
tsv_path = project_dir / "input" / "chunks_pptx.tsv"

# Load 1 chunk with an image_path
import csv

# Load 1 chunk with an image_path using CSV reader for TSV
with open(tsv_path, encoding="utf-8") as f:
    reader = csv.reader(f, delimiter="\t")
    header = next(reader)
    for row in reader:
        if len(row) < 5:
            continue  # skip malformed rows
        meta_json = json.loads(row[4])
        if meta_json.get("image_path"):
            image_chunk = Chunk(
                id=row[0], doc_id=row[1], text=row[2], token_count=int(row[3]), meta=meta_json
            )
            break


if not image_chunk:
    print("❌ No chunk found with image_path.")
    exit()

print(f"✅ Testing chunk from: {image_chunk.meta.get('source_filepath')}")
print(f"    image_path: {image_chunk.meta.get('image_path')}")

# === Run the agent ===
agent = ImageInsightAgent()
project = ProjectManager(project_dir)

result_chunk = agent.run(image_chunk, project)

# === Output ===
print("\n--- Agent Output ---")
if "image_summary" in result_chunk.meta:
    print(result_chunk.meta["image_summary"])
else:
    print("❌ No summary was generated.")
    if "image_summary_error" in result_chunk.meta:
        print("Error:", result_chunk.meta["image_summary_error"])
