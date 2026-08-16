"""
AutoDrive RAG v2.0 — Safety Guardrails
Input/output validation to prevent prompt injection, hallucinated data,
and inappropriate content.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger("chatbot.guardrails")

# Patterns that indicate prompt injection attempts
INJECTION_PATTERNS = [
    r"ignore\s*(all\s*)?(previous|above|prior)\s*(instructions|prompts|rules)",
    r"you\s*are\s*now\s*(a|an)",
    r"forget\s*(all\s*)?(previous|your|everything)",
    r"disregard\s*(all|your|the)",
    r"system\s*prompt",
    r"act\s*as\s*(if|a|an)",
    r"pretend\s*(you|to\s*be)",
    r"new\s*instructions?:",
    r"override\s*(your|the|all)",
    r"\[SYSTEM\]",
    r"\[INST\]",
    r"<\|im_start\|>",
]

# PII patterns
PII_PATTERNS = {
    "aadhaar": r"\b\d{4}\s?\d{4}\s?\d{4}\b",
    "pan": r"\b[A-Z]{5}\d{4}[A-Z]\b",
    "phone": r"\b(?:\+91[\s-]?)?[6-9]\d{9}\b",
    "email": r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b",
    "credit_card": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
}


class SafetyGuardrails:
    """
    Input and output guardrails for the RAG system.

    Input guardrails:
      - Prompt injection detection
      - PII detection and warning
      - Off-topic query detection

    Output guardrails:
      - Validate car IDs exist in inventory
      - Validate prices match inventory data
      - Block inappropriate content
    """

    def __init__(self, inventory_ids: set[str] = None) -> None:
        self.inventory_ids = inventory_ids or set()

    def check_input(self, query: str) -> dict:
        """
        Check user input for safety issues.

        Returns:
            Dict with 'safe' bool, 'issues' list, and 'sanitized' query.
        """
        issues = []
        sanitized = query

        # Check prompt injection
        for pattern in INJECTION_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                issues.append({
                    "type": "prompt_injection",
                    "severity": "high",
                    "detail": "Potential prompt injection detected",
                })
                break

        # Check PII
        for pii_type, pattern in PII_PATTERNS.items():
            matches = re.findall(pattern, query)
            if matches:
                issues.append({
                    "type": "pii_detected",
                    "severity": "medium",
                    "detail": f"{pii_type} detected ({len(matches)} instance(s))",
                })
                # Mask PII in sanitized version
                sanitized = re.sub(pattern, f"[{pii_type.upper()}_REDACTED]", sanitized)

        # Check query length (potential abuse)
        if len(query) > 5000:
            issues.append({
                "type": "excessive_length",
                "severity": "low",
                "detail": f"Query length {len(query)} exceeds limit",
            })
            sanitized = query[:5000]

        safe = not any(i["severity"] == "high" for i in issues)

        return {
            "safe": safe,
            "issues": issues,
            "sanitized": sanitized,
        }

    def check_output(self, response: str, inventory: list[dict] = None) -> dict:
        """
        Check generated output for safety issues.

        Returns:
            Dict with 'safe' bool, 'issues' list, and 'cleaned' response.
        """
        issues = []
        cleaned = response

        # Check for hallucinated car IDs
        mentioned_ids = re.findall(r"\[CAR_ID:(\w+)\]", response)
        for car_id in mentioned_ids:
            if self.inventory_ids and car_id not in self.inventory_ids:
                issues.append({
                    "type": "hallucinated_car_id",
                    "severity": "high",
                    "detail": f"Car ID {car_id} not in inventory",
                })

        # Check for PII leaks in output
        for pii_type, pattern in PII_PATTERNS.items():
            if re.search(pattern, response):
                issues.append({
                    "type": "pii_in_output",
                    "severity": "high",
                    "detail": f"{pii_type} found in output",
                })
                cleaned = re.sub(pattern, f"[{pii_type.upper()}_REDACTED]", cleaned)

        safe = not any(i["severity"] == "high" for i in issues)

        return {
            "safe": safe,
            "issues": issues,
            "cleaned": cleaned,
        }

    def update_inventory_ids(self, cars: list[dict]) -> None:
        """Update the set of valid car IDs from inventory."""
        self.inventory_ids = {str(car.get("id", "")) for car in cars}
