"""User-facing error classification and redaction helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PublicError:
    category: str
    message: str


SECRET_PATTERNS = [
    re.compile(r"\b(?:sk|sk-proj|sk-ant|AIza)[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z0-9 ]*PRIVATE KEY-----"),
    re.compile(r"(?i)\b(https?://)([^/\s:@]+):([^@\s/]+)@"),
    re.compile(r"(?i)\b((?:api[-_]?key|auth[-_]?token|access[-_]?token|token|password|secret)=)[^&\s,;'\"]+"),
    re.compile(r"(?i)\b(authorization\s*:\s*bearer\s+)[^\s,;]+"),
    re.compile(r"(?i)\b(api[-_ ]?key|token|secret|password)(\s*[=:]\s*)[^\s,;'\"]+"),
    re.compile(r"(?i)\b((?:x-api-key|x-goog-api-key|proxy-authorization|api-key)\s*:\s*)[^\s,;]+"),
]


def public_exception(exc: BaseException) -> PublicError:
    message = sanitize_error_text(str(exc) or exc.__class__.__name__)
    return PublicError(category=classify_exception(exc), message=message)


def public_exception_message(exc: BaseException, *, prefix: str) -> str:
    error = public_exception(exc)
    return f"{prefix} ({error.category}): {error.message}"


def sanitize_error_text(text: str) -> str:
    sanitized = text
    for pattern in SECRET_PATTERNS:
        if pattern.groups >= 3:
            sanitized = pattern.sub(lambda match: f"{match.group(1)}[redacted]@", sanitized)
        elif pattern.groups >= 2:
            sanitized = pattern.sub(lambda match: f"{match.group(1)}{match.group(2)}[redacted]", sanitized)
        elif pattern.groups == 1:
            sanitized = pattern.sub(lambda match: f"{match.group(1)}[redacted]", sanitized)
        else:
            sanitized = pattern.sub("[redacted]", sanitized)
    return sanitized


def classify_exception(exc: BaseException) -> str:
    status = _status_code(exc)
    if status in {401, 403}:
        return "auth"
    if status == 429:
        return "rate_limit"
    if status is not None and status >= 500:
        return "service"
    if isinstance(exc, TimeoutError):
        return "timeout"
    name = exc.__class__.__name__.lower()
    text = str(exc).lower()
    if "timeout" in name or "timeout" in text or "timed out" in text:
        return "timeout"
    if any(marker in name for marker in ("connection", "network", "http")):
        return "network"
    if "connection" in text or "network" in text or "temporarily unavailable" in text:
        return "network"
    if "rate limit" in text or "too many requests" in text:
        return "rate_limit"
    if "unauthorized" in text or "forbidden" in text or "api key" in text:
        return "auth"
    return "unknown"


def _status_code(exc: BaseException) -> int | None:
    for attr in ("status_code", "status", "code"):
        value: Any = getattr(exc, attr, None)
        try:
            status = int(value)
        except (TypeError, ValueError):
            continue
        if 100 <= status <= 599:
            return status
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    try:
        status = int(value)
    except (TypeError, ValueError):
        return None
    return status if 100 <= status <= 599 else None
