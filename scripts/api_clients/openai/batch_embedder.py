import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

from openai import OpenAI

# from scripts.utils.logger import LoggerManager


class BatchEmbedder:
    """
    Submits a large embedding job to OpenAI's asynchronous /v1/batches API.
    """

    def __init__(
        self, model: str, output_dir: Path, api_key: Optional[str] = None, logger=None
    ):
        print("\n" + "=" * 100)
        print("DEBUG: BatchEmbedder.__init__() STARTING")
        print("=" * 100)

        self.model = model
        self.output_dir = Path(output_dir)
        self.logger = logger or logging.getLogger(__name__)
        # self.logger = LoggerManager.get_logger("batch_embedder")
        self.api_key = api_key or os.getenv("OPEN_AI")

        print("DEBUG: BatchEmbedder init parameters:")
        print(f"DEBUG:   - model: {self.model}")
        print(f"DEBUG:   - output_dir: {self.output_dir}")
        print(f"DEBUG:   - api_key provided: {bool(api_key)}")
        print(f"DEBUG:   - OPEN_AI env var: {bool(os.getenv('OPEN_AI'))}")
        print(f"DEBUG:   - final api_key: {bool(self.api_key)}")

        if self.api_key:
            print(f"DEBUG:   - api_key starts with: {self.api_key[:10]}...")
        else:
            print("DEBUG:   - NO API KEY FOUND!")

        if not self.api_key:
            error_msg = (
                "API key not found in config or environment variable 'OPEN_AI'"
            )
            print(f"ERROR: {error_msg}")
            raise ValueError(error_msg)

        print("DEBUG: Creating OpenAI client...")
        self.client = OpenAI(api_key=self.api_key)
        print(f"DEBUG: OpenAI client created: {self.client}")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"DEBUG: Output directory created/verified: {self.output_dir}")

        print("=" * 100)
        print("DEBUG: BatchEmbedder.__init__() COMPLETE")
        print("=" * 100)

    def run(
        self, texts: List[str], ids: Optional[List[str]] = None
    ) -> Dict[str, List[float]]:
        print("\n" + "=" * 120)
        print("DEBUG: BatchEmbedder.run() *** MAIN ENTRY POINT ***")
        print("=" * 120)

        print("DEBUG: BatchEmbedder.run() called with:")
        print(f"DEBUG:   - texts count: {len(texts)}")
        print(f"DEBUG:   - ids provided: {bool(ids)}")
        print(f"DEBUG:   - model: {self.model}")
        print(f"DEBUG:   - output_dir: {self.output_dir}")

        if ids is None:
            ids = [f"chunk-{i}" for i in range(len(texts))]
            print(f"DEBUG: Generated {len(ids)} automatic IDs")
        else:
            print(f"DEBUG: Using provided {len(ids)} IDs")

        assert len(texts) == len(ids), "texts and ids must be the same length"
        print("DEBUG: Text/ID length validation passed")

        print("DEBUG: About to prepare JSONL file...")
        input_path = self._prepare_jsonl_file(texts, ids)
        print(f"DEBUG: JSONL file prepared at: {input_path}")

        output_path = (
            self.output_dir / f"openai_batch_{int(time.time())}_output.jsonl"
        )
        print(f"DEBUG: Output path will be: {output_path}")

        print("DEBUG: About to upload file to OpenAI and create batch...")
        print("DEBUG: *** THIS IS WHERE OPENAI API CALLS START ***")

        try:
            with open(input_path, "rb") as f:
                print("DEBUG: File opened for upload")
                print("DEBUG: Calling client.files.create()...")

                # Upload file first
                file_upload = self.client.files.create(file=f, purpose="batch")
                print(f"DEBUG: File uploaded successfully - file_id: {file_upload.id}")

                print("DEBUG: Calling client.batches.create()...")
                batch = self.client.batches.create(
                    input_file_id=file_upload.id,
                    endpoint="/v1/embeddings",
                    completion_window="24h",
                    metadata={
                        "description": f"Batch embedding job with {len(texts)} texts"
                    },
                )
                print(f"DEBUG: Batch created successfully - batch_id: {batch.id}")
                print(f"DEBUG: Batch status: {batch.status}")

                self.logger.info(
                    f"[OPENAI SUBMIT] Submitted async batch job: {batch.id} | "
                    f"status: {batch.status}"
                )
                self.logger.info(
                    f"[OPENAI SUBMIT] Input file: {input_path.name} | "
                    f"Size: {len(texts)} chunks"
                )

        except Exception as e:
            print(f"ERROR: Failed to create batch job: {e}")
            print(f"ERROR: Exception type: {type(e)}")
            raise

        self.logger.info(f"Submitted OpenAI batch job: {batch.id}")
        print("DEBUG: Starting to wait for batch completion...")

        batch = self._wait_for_completion(batch.id)
        print(f"DEBUG: Batch completion wait finished with status: {batch.status}")

        if batch.status != "completed":
            error_msg = f"Batch job failed with status: {batch.status}"
            print(f"ERROR: {error_msg}")
            raise RuntimeError(error_msg)

        print("DEBUG: Batch completed successfully, downloading results...")
        self._download_result_file(batch.output_file_id, output_path)

        print("DEBUG: Loading and parsing output file...")
        result = self._load_output_file(output_path)

        print(f"DEBUG: BatchEmbedder.run() returning {len(result)} embeddings")
        print("=" * 120)
        print("DEBUG: BatchEmbedder.run() *** COMPLETE ***")
        print("=" * 120)

        return result

    def _prepare_jsonl_file(self, texts: List[str], ids: List[str]) -> Path:
        print(f"DEBUG: _prepare_jsonl_file() called with {len(texts)} texts")

        temp_path = self.output_dir / f"batch_input_{int(time.time())}.jsonl"
        print(f"DEBUG: Creating JSONL file at: {temp_path}")

        with open(temp_path, "w", encoding="utf-8") as f:
            for i, text in enumerate(texts):
                # Create proper OpenAI batch format
                request = {
                    "custom_id": ids[i],
                    "method": "POST",
                    "url": "/v1/embeddings",
                    "body": {
                        "model": self.model,
                        "input": text,
                        "encoding_format": "float"
                    },
                }
                f.write(json.dumps(request) + "\n")

                if i < 3:  # Show first few entries for debugging
                    print(f"DEBUG: JSONL entry {i}: {json.dumps(request)}")

        print(f"DEBUG: Wrote {len(texts)} requests to JSONL file")
        self.logger.info(f"Wrote input JSONL file to {temp_path}")
        return temp_path

    def _wait_for_completion(self, batch_id: str) -> "OpenAI.Batch":
        print(f"DEBUG: _wait_for_completion() starting for batch_id: {batch_id}")
        self.logger.info(f"Waiting for batch {batch_id} to complete...")

        iteration = 0
        while True:
            iteration += 1
            print(f"DEBUG: Polling iteration {iteration} for batch {batch_id}")

            try:
                batch = self.client.batches.retrieve(batch_id)
                print(f"DEBUG: Batch status: {batch.status}")
                self.logger.info(f"Batch status: {batch.status}")

                if batch.status in ("completed", "failed", "expired", "cancelled"):
                    print(f"DEBUG: Batch finished with final status: {batch.status}")
                    break

                print("DEBUG: Batch still in progress, sleeping 5 seconds...")
                time.sleep(5)

            except Exception as e:
                print(f"ERROR: Error checking batch status: {e}")
                raise

        return batch

    def _download_result_file(self, file_id: str, output_path: Path) -> None:
        print(f"DEBUG: _download_result_file() called with file_id: {file_id}")
        print(f"DEBUG: Downloading to: {output_path}")

        try:
            content = self.client.files.content(file_id)
            with open(output_path, "wb") as f:
                f.write(content.content)
            print(
                f"DEBUG: Successfully downloaded {output_path.stat().st_size} bytes"
            )
            self.logger.info(f"[OPENAI DOWNLOAD] Downloaded output to: {output_path}")

        except Exception as e:
            print(f"ERROR: Failed to download result file: {e}")
            raise

    def _load_output_file(self, path: Path) -> Dict[str, List[float]]:
        print(f"DEBUG: _load_output_file() called with path: {path}")

        result = {}
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)

                    if i < 3:  # Show first few entries for debugging
                        print(f"DEBUG: Output line {i}: {json.dumps(row, indent=2)}")

                    # Extract embedding from OpenAI batch response format
                    if row.get("response", {}).get("status_code") == 200:
                        custom_id = row["custom_id"]
                        embedding = row["response"]["body"]["data"][0]["embedding"]
                        result[custom_id] = embedding

                        if i < 3:
                            print(
                                f"DEBUG: Extracted embedding for {custom_id}: "
                                f"{len(embedding)} dimensions"
                            )
                    else:
                        print(f"DEBUG: WARNING - Non-200 status for line {i}: {row}")

                except json.JSONDecodeError as e:
                    self.logger.error(
                        f"[ERROR] Failed to parse line {i} in {path.name}"
                    )
                    self.logger.error(f"[ERROR] Raw line: {line[:200]}...")
                    print(f"ERROR: JSON decode error on line {i}: {e}")
                    print(f"ERROR: Raw line: {line[:200]}...")
                    raise e

        print(f"DEBUG: Loaded {len(result)} embeddings from result file")
        self.logger.info(
            f"[OPENAI LOAD] Loaded {len(result)} embeddings from result file"
        )
        return result
