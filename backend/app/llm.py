"""
Thin wrapper around Groq's free-tier chat completion API.

Handles two failure modes that free-tier usage runs into in practice:
- RateLimitError (429): retried with the backoff Groq's own error message
  tells you to use.
- NotFoundError (404): the configured model was decommissioned — falls back
  to GROQ_FALLBACK_MODEL once.
"""
import logging
import re
import time

from groq import Groq, NotFoundError, RateLimitError
from .config import GROQ_API_KEY, GROQ_MODEL, GROQ_FALLBACK_MODEL

logger = logging.getLogger("uvicorn.error")

_client = None

MAX_RATE_LIMIT_RETRIES = 3
DEFAULT_RETRY_SECONDS = 2.0
MAX_RETRY_SECONDS = 10.0

_RETRY_MS_RE = re.compile(r"try again in ([\d.]+)ms")
_RETRY_S_RE = re.compile(r"try again in ([\d.]+)s")


def _get_client() -> Groq:
    global _client
    if _client is None:
        if not GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Copy backend/.env.example to backend/.env "
                "and add your free key from https://console.groq.com/keys"
            )
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


def _retry_delay_seconds(exc: RateLimitError) -> float:
    """Groq puts the exact wait time in the error message (and sometimes a
    Retry-After header); fall back to a fixed default if neither is present."""
    header = None
    response = getattr(exc, "response", None)
    if response is not None:
        header = response.headers.get("retry-after")
    if header:
        try:
            return min(float(header), MAX_RETRY_SECONDS)
        except ValueError:
            pass

    message = str(exc)
    if match := _RETRY_MS_RE.search(message):
        return min(float(match.group(1)) / 1000, MAX_RETRY_SECONDS)
    if match := _RETRY_S_RE.search(message):
        return min(float(match.group(1)), MAX_RETRY_SECONDS)

    return DEFAULT_RETRY_SECONDS


def chat(prompt: str, system: str = "", temperature: float = 0.3, model: str | None = None) -> str:
    """Single-turn chat completion. Returns the assistant's text reply."""
    client = _get_client()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    target_model = model or GROQ_MODEL
    tried_fallback = False
    rate_limit_attempts = 0

    while True:
        try:
            resp = client.chat.completions.create(
                model=target_model,
                messages=messages,
                temperature=temperature,
            )
            return resp.choices[0].message.content
        except NotFoundError:
            if tried_fallback or target_model == GROQ_FALLBACK_MODEL:
                raise
            logger.warning(
                "Groq model '%s' not found (likely decommissioned) — falling back to '%s'",
                target_model,
                GROQ_FALLBACK_MODEL,
            )
            target_model = GROQ_FALLBACK_MODEL
            tried_fallback = True
        except RateLimitError as exc:
            rate_limit_attempts += 1
            if rate_limit_attempts > MAX_RATE_LIMIT_RETRIES:
                raise
            delay = _retry_delay_seconds(exc)
            logger.warning(
                "Groq rate limit on '%s' — retrying in %.2fs (attempt %d/%d)",
                target_model,
                delay,
                rate_limit_attempts,
                MAX_RATE_LIMIT_RETRIES,
            )
            time.sleep(delay)
