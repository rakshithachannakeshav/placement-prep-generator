"""
Free, keyless web search for the Research agent, via DuckDuckGo (`ddgs`).
No API key, no cost — this is what keeps the whole pipeline free-tier.
Note: DuckDuckGo can rate-limit rapid repeated queries; the graph only
issues a couple of searches per report run, which stays well within that.
"""
import logging

from ddgs import DDGS

logger = logging.getLogger("uvicorn.error")

SEARCH_TIMEOUT_SECONDS = 8


def search(query: str, max_results: int = 5) -> list[dict]:
    """Returns a list of {title, href, body} dicts — or [] if search fails.
    Company research is a nice-to-have; it shouldn't take the whole report
    down when a free, best-effort search API has a bad day. Deliberately
    catches broadly: ddgs wraps most backend failures in DDGSException, but
    some (e.g. httpx's own TimeoutException — a different class from ddgs's
    identically-named one) slip through unwrapped, and the point of this
    function is that no search-layer failure should ever propagate."""
    try:
        with DDGS(timeout=SEARCH_TIMEOUT_SECONDS) as ddgs:
            return list(ddgs.text(query, max_results=max_results))
    except Exception as exc:
        logger.warning("Web search failed for %r: %s: %s", query, type(exc).__name__, exc)
        return []


def search_as_context(query: str, max_results: int = 5, prefer_term: str | None = None) -> str:
    """Search and format results as a single text blob for LLM context.

    Short/generic queries (company names like "Q2") pull in unrelated results
    from other companies. If `prefer_term` is given, results are filtered to
    those where it appears in the title or body — falling back to the
    unfiltered set only if that filter would leave nothing.
    """
    results = search(query, max_results=max_results)
    if not results:
        return "No web search results found."

    if prefer_term:
        needle = prefer_term.lower()
        filtered = [
            r for r in results
            if needle in r.get("title", "").lower() or needle in r.get("body", "").lower()
        ]
        if filtered:
            results = filtered

    lines = []
    for r in results:
        lines.append(f"- {r.get('title', '')}: {r.get('body', '')} ({r.get('href', '')})")
    return "\n".join(lines)
