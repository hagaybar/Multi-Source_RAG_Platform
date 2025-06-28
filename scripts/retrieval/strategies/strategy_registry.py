# scripts/retrieval/strategy_registry.py

from scripts.retrieval.strategies.late_fusion import late_fusion

# from scripts.retrieval.strategies.agentic_routing import agentic_routing  # future

STRATEGY_REGISTRY = {
    "late_fusion": late_fusion,
    # "agentic": agentic_routing
}
