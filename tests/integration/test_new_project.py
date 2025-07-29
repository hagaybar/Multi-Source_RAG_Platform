import sys
from pathlib import Path
import shutil
import yaml
from scripts.core.project_manager import ProjectManager


def test_create_project(tmp_path):
    # Setup
    projects_base_dir = tmp_path / "projects"
    projects_base_dir.mkdir()
    project_name = "test_project"
    project_description = "A test project"
    language = "en"
    image_enrichment = True
    embedding_model = "text-embedding-ada-002"

    # Create project
    ProjectManager.create_project(
        project_name=project_name,
        project_description=project_description,
        language=language,
        image_enrichment=image_enrichment,
        embedding_model=embedding_model,
        projects_base_dir=projects_base_dir,
    )

    # Verify
    project_root = projects_base_dir / project_name
    assert project_root.exists()
    assert (project_root / "config.yml").exists()
    assert (project_root / "input" / "raw").exists()
    assert (project_root / "output" / "logs").exists()
    assert (project_root / "output" / "faiss").exists()
    assert (project_root / "output" / "metadata").exists()

    with (project_root / "config.yml").open("r") as f:
        config = yaml.safe_load(f)

    assert config["project"]["name"] == project_name
    assert config["project"]["description"] == project_description
    assert config["project"]["language"] == language
    assert config["project"]["image_enrichment"] == image_enrichment
    assert config["embedding"]["model"] == embedding_model
