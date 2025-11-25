#!/usr/bin/env python3
"""Simple Phase 1 pipeline run - no fancy testing, just reprocess."""
import sys
from pathlib import Path
from scripts.core.project_manager import ProjectManager
from scripts.pipeline.runner import PipelineRunner

project_path = Path("data/projects/Primo_List")
project = ProjectManager(project_path)
runner = PipelineRunner(project, config=project.config)

print("Starting Phase 1 pipeline...")
print("Step 1: Ingest (with Phase 1 cleaning)")
runner.add_step("ingest")

print("Step 2: Chunk (with semantic chunking)")
runner.add_step("chunk")

print("\nRunning pipeline...")
for msg in runner.run_steps():
    print(msg)

print("\n✅ Done! Ready for embedding.")
print("\nTo embed, run in UI or add runner.add_step('embed')")
