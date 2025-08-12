import csv


def load_csv(file_path: str) -> tuple[str, dict]:
    """
    Loads a CSV file, concatenates all rows into a single string,
    and returns the string along with metadata.

    Args:
        file_path: The path to the CSV file.

    Returns:
        A tuple containing the full CSV text (str) and metadata (dict).
    """
    full_csv_text = ""
    try:
        with open(file_path, 'r', newline='') as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                full_csv_text += ",".join(row) + "\n"
    except FileNotFoundError:
        # Or handle more gracefully, e.g., by raising a custom exception
        # or returning an error message. For now, let's assume the file exists
        # or this is handled by the caller.
        raise
    except Exception as e:
        # Handle other potential CSV reading errors
        print(f"Error reading CSV file {file_path}: {e}")
        raise

    metadata_dict = {'doc_type': 'csv'}
    return full_csv_text, metadata_dict


if __name__ == '__main__':
    # Example usage:
    # Create a dummy CSV file for testing
    dummy_csv_content = """Header1,Header2,Header3
r1c1,r1c2,r1c3
r2c1,r2c2,r2c3
r3c1,r3c2,r3c3
"""
    dummy_file_path = "test.csv"
    with open(dummy_file_path, 'w') as f:
        f.write(dummy_csv_content)

    text, metadata = load_csv(dummy_file_path)
    print("------- Full CSV Text -------")
    print(text)
    print("--------- Metadata ----------")
    print(metadata)

    # Clean up the dummy file
    import os

    os.remove(dummy_file_path)

    # Test with a non-existent file
    try:
        load_csv("non_existent.csv")
    except FileNotFoundError:
        print("\nSuccessfully caught FileNotFoundError for non_existent.csv")
