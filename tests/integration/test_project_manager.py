import sys
from pathlib import Path
import shutil
from scripts.core.project_manager import ProjectManager


def test_project_manager_initialization(tmp_path):
    # Setup: copy minimal config into temp project folder
    project_root = tmp_path / "demo_project"
    project_root.mkdir()
    (project_root / "config.yml").write_text("""
project:
  name: demo_project
paths:
  input_dir: input
  output_dir: output
  logs_dir: output/logs
  faiss_dir: output/faiss
  metadata_dir: output/metadata
""")

    pm = ProjectManager(project_root)

    assert pm.input_dir.exists()
    assert pm.output_dir.exists()
    assert pm.logs_dir.exists()
    assert pm.get_faiss_path("pdf").name == "pdf.faiss"
    assert pm.config["project"]["name"] == "demo_project"
    print(f"Temp project created at: {tmp_path}")

    print("ProjectManager is working ✔")
