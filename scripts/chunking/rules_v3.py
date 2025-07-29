import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Dict


@dataclass
class ChunkRule:
    strategy: str
    # e.g., [200, 800] or from min_chunk_size
    min_tokens: int
    max_tokens: int
    overlap: int


_rules_data = None
_rules_file_path = Path(__file__).parent.parent.parent / "configs" / "chunk_rules.yaml"

REQUIRED_KEYS = {"strategy", "min_tokens", "max_tokens", "overlap"}


def _load_rules_if_needed():
    """Read configs/chunk_rules.yaml, validate, and cache ChunkRule objects."""
    global _rules_data
    if _rules_data is not None:
        return  # already loaded
    else:  # Loads rules from YAML if not already loaded."""
        # -------- locate file ---------------------------------------------------
        path_to_try = _rules_file_path  # e.g. <repo>/scripts/../configs/chunk_rules.yaml
        if not path_to_try.exists():
            # Fallback: cwd/configs/chunk_rules.yaml  (useful when running tests)
            path_to_try = Path("configs") / "chunk_rules.yaml"
            if not path_to_try.exists():
                raise FileNotFoundError(
                    f"Rules file not found at {_rules_file_path} or {path_to_try}"
                )

        # -------- load YAML -----------------------------------------------------
        with path_to_try.open(encoding="utf-8") as fh:
            raw_rules: Dict[str, Dict] = yaml.safe_load(fh) or {}
        # -------- validate & convert -------------------------------------------
        validated: Dict[str, ChunkRule] = {}
        for doc_type, rule in raw_rules.items():
            missing = REQUIRED_KEYS - rule.keys()
            if missing:
                raise ValueError(f"{doc_type}: missing keys {missing} in chunk_rules.yaml")

            # Build dataclass; extra YAML fields (e.g., comments or 'notes') are ignored
            validated[doc_type] = ChunkRule(
                strategy=rule["strategy"],
                min_tokens=int(rule["min_tokens"]),
                max_tokens=int(rule["max_tokens"]),
                overlap=int(rule["overlap"]),
            )
            # sanity: ensure min < max
            if validated[doc_type].min_tokens >= validated[doc_type].max_tokens:
                raise ValueError(
                    f"{doc_type}: min_tokens must be < max_tokens "
                    f"({validated[doc_type].min_tokens} ≥ {validated[doc_type].max_tokens})"
                )

        _rules_data = validated


# scripts/chunking/rules_v3.py
def get_rule(doc_type: str) -> ChunkRule:
    """Return the ChunkRule for the given doc_type (or fallback to 'default')."""
    _load_rules_if_needed()

    rule = _rules_data.get(doc_type)
    if rule is None:
        rule = _rules_data["default"]

    # rule is already a ChunkRule object, not a dict
    return rule


def get_all_rules() -> dict[str, ChunkRule]:
    """
    Retrieves all chunking rules.

    Returns:
        A dictionary mapping document types to ChunkRule objects.

    Raises:
        FileNotFoundError: If the rules YAML file is not found.
    """
    _load_rules_if_needed()
    assert _rules_data is not None, "Rules data should be loaded"

    all_chunk_rules = {}
    for doc_type, rule_dict in _rules_data.items():
        all_chunk_rules[doc_type] = ChunkRule(**rule_dict)
    return all_chunk_rules


def get_all_doc_types() -> list[str]:
    """
    Retrieves all document types defined in the chunking rules.

    Returns:
        A list of document type strings.

    Raises:
        FileNotFoundError: If the rules YAML file is not found.
    """
    rules = get_all_rules()
    return list(rules.keys())
