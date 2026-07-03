import re
from typing import Optional


def clean_text(text: Optional[str]) -> str:
    """Normalize text by removing noise and extra whitespace."""
    if not text:
        return ""

    normalized = text.lower().strip()
    normalized = re.sub(r"[^a-z0-9\s\-\.\$%]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized
