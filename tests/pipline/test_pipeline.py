from pathlib import Path
import yaml
from scripts.core.project_manager import ProjectManager
from scripts.pipeline.runner import PipelineRunner


def main():
    # Load config and project
    project_root = Path("data/projects/test_minimal_data")
    config_path = project_root / "config.yml"

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    project = ProjectManager(project_root)
    runner = PipelineRunner(project, config)

    # Add steps: ingest → chunk
    runner.clear_steps()
    runner.add_step("ingest")
    runner.add_step("chunk")

    # runner.add_step("enrich")
    
    # Run and stream output
    for msg in runner.run_steps():
        print(msg)


if __name__ == "__main__":
    main()
