from pathlib import Path
from scripts.utils.image_utils import get_project_image_dir, infer_project_root

somepath = Path(
    r"Multi-Source_RAG_Platform/data/projects/test_coordinators_data/"
    r"input/raw/ACQ/ELUNA 2016 - New and Emerging Acquisition "
    r"Workflows - Purchase Requests - Yoel Kortick.pdf"
)

root = infer_project_root(somepath)
project_name = root.name  # ✅ get the last part of the path
image_dir = get_project_image_dir(project_name)
print(f"Project root: {root}")
print(f"Project name: {project_name}")

print(get_project_image_dir(image_dir))
