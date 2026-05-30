import re
import hashlib
from typing import Optional, List, Any
from phoenix.core.config import config
from phoenix.services.observability.logger import get_logger

logger = get_logger("Phoenix AI.Security")

class SecurityError(Exception):
    """Base exception for framework security violations."""
    pass

class SecurityGuard:
    """
    Framework-level security orchestrator.
    Handles input validation, injection detection, and output sanitization.
    """
    def __init__(self, mode: str = "standard"):
        self.mode = mode
        self.max_length = config.SECURITY_MAX_INPUT_LENGTH
        
        # Common prompt injection patterns
        self.injection_patterns = [
            r"ignore previous instructions",
            r"ignore all rules",
            r"system:",
            r"assistant:",
            r"you are now a",
            r"bypass security"
        ]

    async def validate_input(self, text: str) -> str:
        """
        Scan input for potential threats.
        - DOS: Length validation
        - Prompt Injection: Pattern matching
        """
        if not text:
            return ""

        # 1. DOS Protection: Length Check
        if len(text) > self.max_length:
            logger.warning(f"SECURITY ALERT: Input length {len(text)} exceeds limit {self.max_length}")
            raise SecurityError(f"Input too long. Max allowed: {self.max_length} characters.")

        # 2. Prompt Injection Detection
        text_lower = text.lower()
        for pattern in self.injection_patterns:
            if re.search(pattern, text_lower):
                logger.warning(f"SECURITY ALERT: Potential Prompt Injection detected: '{pattern}'")
                # In standard mode, we might just sanitize or warn. In strict, we block.
                if self.mode == "strict":
                    raise SecurityError("Potential prompt injection detected. Request blocked.")
                else:
                    # Basic sanitization: Wrap the input to reinforce context
                    return f"[Restricted Input]: {text}"

        return text

    def mask_secrets(self, text: str) -> str:
        """
        Redact sensitive information from strings (like API keys).
        """
        if not text:
            return ""
        
        # Regex for generic API keys (sk-..., ak-..., etc.)
        # Typical format: 20-50 alphanumeric chars
        patterns = [
            (r"(sk-[a-zA-Z0-9]{20,})", "sk-REDACTED"),
            (r"(ak-[a-zA-Z0-9]{20,})", "ak-REDACTED"),
            (r"([a-f0-9]{32})", "HASH-REDACTED") # Generic 32-char hex (like secret keys)
        ]
        
        masked_text = text
        for pattern, replacement in patterns:
            masked_text = re.sub(pattern, replacement, masked_text)
            
        return masked_text

    async def verify_grounding(self, response: str, context: str) -> bool:
        """
        Heuristic check for hallucination.
        Verifies if the response key terms are present in the provided context.
        """
        if not config.SECURITY_ENABLE_HALLUCINATION_CHECK or not context:
            return True
            
        # Basic check: extract significant words from response and check context
        # This is a placeholder for more advanced NLI or distance-based checks
        response_words = set(re.findall(r"\w{5,}", response.lower()))
        context_lower = context.lower()
        
        matches = [word for word in response_words if word in context_lower]
        coverage = len(matches) / len(response_words) if response_words else 1.0
        
        if coverage < 0.3: # Threshold
            logger.warning(f"SECURITY WARNING: Low grounding detected ({coverage:.2f}). Potential hallucination.")
            return False
            
        return True

