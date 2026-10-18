"""Web search. DuckDuckGo through the `ddgs` package (no key, no account), or your own
SearXNG instance if pod.yaml sets tools.searxng_url (fully self-hosted)."""

from __future__ import annotations

import os
from urllib.parse import urlparse

from . import Tool


def _fmt(results: list[dict], query: str) -> str:
    if not results:
        return f"No web results for: {query}"
    out = [f"Web results for: {query}"]
    for i, r in enumerate(results, 1):
        host = urlparse(r.get("url") or "").netloc.replace("www.", "")
        date = f" ({r['date'][:10]})" if r.get("date") else ""
        out.append(f"{i}. {r.get('title', '').strip()}{date} [{host}]\n   {(r.get('body') or '').strip()[:400]}")
    return "\n".join(out)


def search(query: str, max_results: int = 5, searxng_url: str | None = None, news: bool = False) -> str:
    if os.environ.get("IRIS_OFFLINE"):
        return "Web search is offline."
    if searxng_url:
        import requests
        r = requests.get(searxng_url.rstrip("/") + "/search",
                         params={"q": query, "format": "json", "categories": "news" if news else "general"},
                         timeout=10)
        r.raise_for_status()
        items = [{"title": x.get("title"), "body": x.get("content"), "url": x.get("url"),
                  "date": x.get("publishedDate")} for x in r.json().get("results", [])[:max_results]]
        return _fmt(items, query)
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS   # older package name
        except ImportError:
            return "Web search isn't installed. Run: ./setup.sh (it installs the 'ddgs' package)."
    import time
    raw = []
    for attempt in range(3):                       # DuckDuckGo sometimes answers a valid query with nothing
        with DDGS(timeout=10) as d:
            try:
                kind = news if attempt != 1 else not news            # second try: the other index
                raw = d.news(query, max_results=max_results) if kind else d.text(query, max_results=max_results)
            except Exception:
                if attempt == 2:
                    raise
                raw = []
        if raw:
            break
        time.sleep(0.6)
    items = [{"title": x.get("title"), "body": x.get("body") or x.get("excerpt"),
              "url": x.get("href") or x.get("url"), "date": x.get("date")} for x in raw or []]
    return _fmt(items, query)


def tool(settings=None) -> Tool:
    cfg = ((settings.pod.get("tools") or {}) if settings is not None else {}) or {}
    url = cfg.get("searxng_url") or os.environ.get("IRIS_SEARXNG")

    def run(args: dict) -> str:
        q = str(args.get("query") or "").strip()
        if not q:
            return "No query given."
        return search(q, int(cfg.get("max_results", 5)), url, news=bool(args.get("news")))

    return Tool(
        name="web_search",
        description="Search the web for current or factual information: news, weather, prices, sports, "
                    "people, recent events, anything you're unsure about. Returns titles and snippets.",
        parameters={"type": "object", "properties": {
            "query": {"type": "string", "description": "what to search for, like you'd type it into a search engine"},
            "news": {"type": "boolean", "description": "true for recent news"}},
            "required": ["query"]},
        run=run, say_first="Let me look that up.", text_form="<<search: QUERY>>", text_arg="query")
