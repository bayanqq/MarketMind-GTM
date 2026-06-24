"""
utils/security.py
-----------------
Prompt injection defense layer and general input validation for MarketMind AI.

All user-supplied text must pass through ``validate_input_text`` before being
forwarded to any LLM, search, or embedding API call.
"""

import re

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_INPUT_LENGTH: int = 4000

# Patterns that indicate an attempt to override or hijack system instructions.
_INJECTION_PATTERNS: list[str] = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"disregard\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"forget\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"you\s+are\s+now\s+(?:a|an)\s+\w+",
    r"act\s+as\s+(?:a|an)\s+(?:unrestricted|unfiltered|jailbroken)",
    r"do\s+anything\s+now",
    r"dan\s+mode",
    r"developer\s+mode",
    r"override\s+(?:safety|content|system)\s+(?:filter|policy|guidelines)",
    r"pretend\s+(?:you\s+have\s+no\s+restrictions|to\s+be\s+unrestricted)",
    r"system\s*:\s*you\s+are",
    r"<\s*system\s*>",
    r"\[system\]",
    r"###\s*instruction",
    r"new\s+instructions\s*:",
    r"print\s+your\s+(system\s+)?prompt",
    r"reveal\s+your\s+(system\s+)?instructions",
    r"output\s+your\s+initial\s+prompt",
]

_COMPILED_PATTERNS: list[re.Pattern] = [
    re.compile(pattern, re.IGNORECASE) for pattern in _INJECTION_PATTERNS
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_input_text(text: str) -> bool:
    """Validate user-supplied text against injection attacks and length limits.

    Args:
        text: The raw string to validate.

    Returns:
        ``True`` when the text is considered safe and within length bounds.
        ``False`` when a violation is detected.

    Raises:
        TypeError: When *text* is not a string.
    """
    if not isinstance(text, str):
        raise TypeError(f"validate_input_text expects str, got {type(text).__name__!r}")

    # ── Length guard ────────────────────────────────────────────────────────
    if len(text) > MAX_INPUT_LENGTH:
        print(
            f"[Security] Input rejected: length {len(text)} exceeds "
            f"maximum {MAX_INPUT_LENGTH} characters."
        )
        return False

    # ── Injection detection ─────────────────────────────────────────────────
    for pattern in _COMPILED_PATTERNS:
        match = pattern.search(text)
        if match:
            print(
                f"[Security] Input rejected: potential prompt injection detected "
                f"(matched: '{match.group(0)}')."
            )
            return False

    return True


def sanitize_for_query(text: str, max_chars: int = 250) -> str:
    """Strip newlines and truncate text to *max_chars* for safe API queries.

    Args:
        text: The string to sanitize.
        max_chars: Hard character ceiling after sanitization.

    Returns:
        A cleaned, length-bounded string suitable for use in a search query.
    """
    sanitized = re.sub(r"[\r\n\t]+", " ", text).strip()
    return sanitized[:max_chars]
