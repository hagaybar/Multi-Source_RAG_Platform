import faiss
from pathlib import Path


def get_faiss_dimensions(faiss_path: str | Path) -> tuple[int, int]:
    """
    Returns (dimension, number_of_vectors) for the FAISS index at given path.

    Args:
        faiss_path: Path to the .faiss file

    Returns:
        A tuple (dimension, count)
    """
    path = Path(faiss_path)
    if not path.exists():
        raise FileNotFoundError(f"FAISS file not found: {faiss_path}")

    index = faiss.read_index(str(path))
    return index.d, index.ntotal


if __name__ == "__main__":
    import sys

    # if len(sys.argv) != 2:
    #     print("Usage: python get_faiss_dimensions.py <path_to_faiss_file>")
    #     sys.exit(1)

    # dim, count = get_faiss_dimensions(sys.argv[1])
    dim, count = get_faiss_dimensions(
        r"C:\git projects\Multi-Source_RAG_Platform\data\projects\demo_project_batch_api\output\faiss\docx.faiss"
    )
    print(f"FAISS index has dimension: {dim}, number of vectors: {count}")
