import os
import json
import time
import tempfile
from pathlib import Path
from typing import List, Dict, Optional
from openai import OpenAI

from scripts.utils.logger import LoggerManager


class BatchEmbedder:
    """
    Submits a large embedding job to OpenAI's asynchronous /v1/batches API.
    """

    def __init__(self, model: str, output_dir: Path, api_key: Optional[str] = None):
        self.model = model
        self.output_dir = Path(output_dir)
        self.logger = LoggerManager.get_logger("batch_embedder")
        self.api_key = api_key or os.getenv("OPEN_AI")

        if not self.api_key:
            raise ValueError("API key not found in config or environment variable 'OPEN_AI'")

        self.client = OpenAI(api_key=self.api_key)

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, texts: List[str], ids: Optional[List[str]] = None) -> Dict[str, List[float]]:
        if ids is None:
            ids = [f"chunk-{i}" for i in range(len(texts))]

        assert len(texts) == len(ids), "texts and ids must be the same length"

        input_path = self._prepare_jsonl_file(texts, ids)
        output_path = self.output_dir / f"openai_batch_{int(time.time())}_output.jsonl"

        with open(input_path, "rb") as f:
            batch = self.client.batches.create(
                input_file=f,
                endpoint="/v1/embeddings",
                parameters={"model": self.model}
            )

        self.logger.info(f"Submitted OpenAI batch job: {batch.id}")
        batch = self._wait_for_completion(batch.id)

        if batch.status != "completed":
            raise RuntimeError(f"Batch job failed with status: {batch.status}")

        self._download_result_file(batch.output_file_id, output_path)
        return self._load_output_file(output_path)

    def _prepare_jsonl_file(self, texts: List[str], ids: List[str]) -> Path:
        temp_path = self.output_dir / f"batch_input_{int(time.time())}.jsonl"
        with open(temp_path, "w", encoding="utf-8") as f:
            for i, text in enumerate(texts):
                f.write(json.dumps({"input": text, "custom_id": ids[i]}) + "\\n")
        self.logger.info(f"Wrote input JSONL file to {temp_path}")
        return temp_path

    def _wait_for_completion(self, batch_id: str) -> "OpenAI.Batch":
        self.logger.info(f"Waiting for batch {batch_id} to complete...")
        while True:
            batch = self.client.batches.retrieve(batch_id)
            self.logger.info(f"Batch status: {batch.status}")
            if batch.status in ("completed", "failed", "expired", "cancelled"):
                break
            time.sleep(5)
        return batch

    def _download_result_file(self, file_id: str, output_path: Path) -> None:
        with open(output_path, "wb") as f:
            self.client.files.download(file_id=file_id, write_to=f)
        self.logger.info(f"Downloaded batch result to {output_path}")

    def _load_output_file(self, path: Path) -> Dict[str, List[float]]:
        with open(path, "r", encoding="utf-8") as f:
            result = {}
            for line in f:
                row = json.loads(line)
                result[row["custom_id"]] = row["embedding"]
        self.logger.info(f"Loaded {len(result)} embeddings from output.")
        return result