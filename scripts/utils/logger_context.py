import logging
from typing import Dict

class ContextLogger(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        kwargs.setdefault("extra", {})
        kwargs["extra"].setdefault("extra_data", {})
        # Merge adapter context into record.extra_data (used by JsonLogFormatter)
        kwargs["extra"]["extra_data"].update(self.extra)
        return msg, kwargs

def with_context(logger: logging.Logger, **ctx: Dict):
    """
    Return a LoggerAdapter that automatically adds context fields
    (e.g., run_id, component) to every log line.
    """
    return ContextLogger(logger, ctx or {})
