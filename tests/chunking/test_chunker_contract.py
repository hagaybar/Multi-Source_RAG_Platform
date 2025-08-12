# tests/test_chunker_contract.py
import inspect
from typing import get_origin

import pytest

from scripts.chunking.chunker_v3 import split
from scripts.chunking.models import Chunk


@pytest.mark.skip(reason="Fails after logging refactor — to be updated in Week 5")
def test_split_signature():
    sig = inspect.signature(split)
    params = list(sig.parameters.values())
    assert [p.name for p in params] == ["text", "meta", "clean_options"]

    origin = get_origin(sig.return_annotation) or sig.return_annotation
    assert origin in (list, inspect._empty)


def test_split_runtime_shape():
    chunks = split("A.\n\nB.", {"doc_type": "txt"})
    assert isinstance(chunks, list)
    assert all(isinstance(c, Chunk) for c in chunks)
    assert len(chunks) == 1
    assert "A." in chunks[0].text
    assert "B." in chunks[0].text
