# Index Module

This directory contains scripts and modules related to the management and inspection of the vector indexes used in the RAG platform. These indexes are essential for efficiently searching and retrieving relevant information.

## Scripts

### `inpect_faiss.py`

This is a utility script for inspecting FAISS index files (`.faiss`). It provides a simple way to get the key properties of an index without having to load it in a larger application.

**Functionality:**

- **`get_faiss_dimensions(faiss_path)`**: This function takes the path to a `.faiss` file and returns a tuple containing:
    - The dimension of the vectors in the index.
    - The total number of vectors stored in the index.

**Usage:**

The script can be run directly from the command line to inspect a FAISS index file:

```bash
python -m scripts.index.inpect_faiss <path_to_faiss_file>
```

This will print the dimension and vector count to the console.
