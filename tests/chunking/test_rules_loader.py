from scripts.chunking.rules_v3 import get_rule, ChunkRule


def test_rule_txt():
    rule = get_rule("txt")
    assert isinstance(rule, ChunkRule)
    assert rule.strategy == "by_paragraph"  # or whatever you set in YAML
    assert rule.min_tokens < rule.max_tokens
