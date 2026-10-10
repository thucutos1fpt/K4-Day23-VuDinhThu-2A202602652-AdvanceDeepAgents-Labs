"""tools.py - STUDENT IMPLEMENTS.  Source tools for the research agents.   Guide: GUIDE.md, part 1.

Rules for every tool:
  * runs on the HOST (not in the sandbox): API keys must never enter the sandbox;
  * returns a STRING (JSON text of compact records) and NEVER raises:
        "NO RESULTS"  when the source answers with nothing,
        "ERROR: ..."  when the source keeps failing after the retries (the agent then tries another source);
  * the docstring is the tool description the LLM reads: keep it precise (what it does, what it returns, when to use it).
Try your tools without any agent:   python tools.py
"""
import json  # noqa: F401
import os  # noqa: F401
import random
import re
import time  # noqa: F401
import xml.etree.ElementTree  # noqa: F401  (arXiv answers with Atom XML)
from urllib.parse import urlencode

import httpx  # noqa: F401
from langchain_core.tools import tool

# ---- constants (given) ----
ARXIV_URL = "https://export.arxiv.org/api/query"  # https only: http answers 301
HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
HF_SEARCH_URL = "https://huggingface.co/api/papers/search"
EXA_URL = "https://mcp.exa.ai/mcp"
_last_arxiv_call = 0.0


class RetryableError(Exception):
    """Given. Raise it inside a call to ask with_retry to wait and try again (retry_after in seconds, optional)."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


# ---- TODO 1: retry helper ----
def with_retry(fn, *, attempts=5, base=1.0, cap=30.0):
    """Call fn(); when it raises RetryableError, wait and call it again.

    PSEUDO-CODE:
      for attempt in 0 .. attempts-1:
          try: return fn()
          except RetryableError as e:
              if this was the last attempt: raise
              delay = e.retry_after if the server told us, else exponential backoff base * 2**attempt
              cap the delay at `cap` seconds; add random jitter to the exponential case
              sleep(delay)
    Use it to wrap EVERY network call below. Also treat these as retryable: HTTP 429/500/502/503/504,
    httpx.TransportError (timeouts, connection resets). Read the Retry-After header when present.
    """
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    for attempt in range(attempts):
        try:
            return fn()
        except RetryableError as exc:
            if attempt == attempts - 1:
                raise
            if exc.retry_after is None:
                delay = min(cap, base * (2**attempt))
                # Jitter prevents simultaneous clients from retrying in lockstep.
                delay += random.uniform(0, min(1.0, delay * 0.2))
            else:
                delay = min(cap, max(0.0, float(exc.retry_after)))
            time.sleep(delay)


def _retry_after(response):
    """Best-effort parsing of the standard Retry-After header."""
    value = response.headers.get("Retry-After")
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def _raise_for_retryable_status(response):
    if response.status_code in {429, 500, 502, 503, 504}:
        raise RetryableError(f"HTTP {response.status_code}", _retry_after(response))
    response.raise_for_status()


def _request(method, url, **kwargs):
    """Make one HTTP request, converting transient HTTP/network failures to RetryableError."""
    try:
        response = getattr(httpx, method)(url, timeout=30.0, **kwargs)
        _raise_for_retryable_status(response)
        return response
    except httpx.TransportError as exc:
        raise RetryableError(f"network error: {exc}") from exc


def _compact(value, limit=600):
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


def _as_int(value, default=0):
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return default


def _error_text(exc, secret=""):
    message = f"{type(exc).__name__}: {exc}"
    return message.replace(secret, "[REDACTED]") if secret else message


# ---- TODO 2: arXiv ----
@tool
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv papers by keywords, newest first. Returns a JSON list of {id, url, published, title, summary}."""
    # PSEUDO-CODE:
    #   keep only word characters of `query` -> terms; no terms -> "NO RESULTS" (do not call the network)
    #   respect arXiv etiquette: at least 3 seconds between two arXiv calls (remember the time of the last call)
    #   GET ARXIV_URL params: search_query="all:t1 AND all:t2 ...", sortBy=submittedDate, sortOrder=descending,
    #       max_results=clamp(max_results, 1, 30)           (wrap in with_retry)
    #   parse the Atom XML: each <entry> -> {id (last part of <id> after /abs/), url, published[:10], title, summary}
    #       collapse whitespace/newlines in title and summary; cut summary to ~600 chars
    #   no entries -> "NO RESULTS"; else json.dumps(records, ensure_ascii=False)
    #   any exception -> "ERROR: <type>: <message>"
    terms = re.findall(r"[\w-]+", query or "", flags=re.UNICODE)
    if not terms:
        return "NO RESULTS"
    limit = max(1, min(_as_int(max_results, 10), 30))

    def fetch():
        global _last_arxiv_call
        remaining = 3.0 - (time.monotonic() - _last_arxiv_call)
        if remaining > 0:
            time.sleep(remaining)
        _last_arxiv_call = time.monotonic()
        return _request("get", ARXIV_URL, params={
            "search_query": " AND ".join(f"all:{term}" for term in terms),
            "sortBy": "submittedDate", "sortOrder": "descending", "max_results": limit,
        })

    try:
        response = with_retry(fetch, attempts=6, cap=60.0)
        root = xml.etree.ElementTree.fromstring(response.content)
        atom = "{http://www.w3.org/2005/Atom}"
        records = []
        for entry in root.iterfind(f"{atom}entry"):
            raw_id = (entry.findtext(f"{atom}id") or "").rsplit("/abs/", 1)[-1]
            paper_id = re.sub(r"v\d+$", "", raw_id)
            if not paper_id:
                continue
            records.append({
                "id": paper_id,
                "url": f"https://arxiv.org/abs/{paper_id}",
                "published": (entry.findtext(f"{atom}published") or "")[:10],
                "title": _compact(entry.findtext(f"{atom}title")),
                "summary": _compact(entry.findtext(f"{atom}summary")),
            })
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:  # Tools must be safe for an agent to call.
        return f"ERROR: {_error_text(exc)}"


# ---- TODO 3: Hugging Face ----
@tool
def hf_daily_papers(limit: int = 30, date: str = "", keyword: str = "") -> str:
    """Hugging Face Daily Papers = what is trending in AI research. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars} sorted by upvotes. `date` is YYYY-MM-DD (empty = latest).
    `keyword` filters title/summary; there is no topic search on this endpoint (use hf_search_papers for a topic)."""
    # PSEUDO-CODE:
    #   GET HF_DAILY_URL params: limit (clamp 1..100) and date (only when given)      (with_retry)
    #   response = list of items {"paper": {id, title, summary, upvotes, githubRepo, githubStars, publishedAt}, ...}
    #   map every item to the record shape above (skip items without paper.id); url = https://huggingface.co/papers/<id>
    #   keyword -> keep records whose title+summary contains it (case-insensitive); sort by upvotes descending
    params = {"limit": max(1, min(_as_int(limit, 30), 100))}
    if date:
        params["date"] = date
    try:
        data = with_retry(lambda: _request("get", HF_DAILY_URL, params=params).json())
        records = _hf_records(data)
        needle = keyword.strip().lower()
        if needle:
            records = [r for r in records if needle in f"{r['title']} {r['summary']}".lower()]
        records.sort(key=lambda item: item["upvotes"], reverse=True)
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:
        return f"ERROR: {_error_text(exc)}"


@tool
def hf_search_papers(query: str, limit: int = 10) -> str:
    """Search Hugging Face papers by topic. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars}."""
    # PSEUDO-CODE:
    #   GET HF_SEARCH_URL params: q=query, limit (clamp 1..50)                         (with_retry)
    #   same item shape as the daily endpoint; prefer paper["ai_summary"] over paper["summary"] when present
    if not (query or "").strip():
        return "NO RESULTS"
    try:
        data = with_retry(lambda: _request("get", HF_SEARCH_URL, params={
            "q": query.strip(), "limit": max(1, min(_as_int(limit, 10), 50)),
        }).json())
        records = _hf_records(data, prefer_ai_summary=True)
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:
        return f"ERROR: {_error_text(exc)}"


def _hf_records(data, prefer_ai_summary=False):
    """Normalize the slightly different paper shapes returned by HF endpoints."""
    if isinstance(data, dict):
        data = data.get("papers") or data.get("data") or []
    records = []
    for item in data if isinstance(data, list) else []:
        paper = item.get("paper", item) if isinstance(item, dict) else {}
        paper_id = paper.get("id")
        if not paper_id:
            continue
        summary = paper.get("ai_summary") if prefer_ai_summary else None
        summary = summary or paper.get("summary") or item.get("summary", "")
        records.append({
            "id": paper_id,
            "url": f"https://huggingface.co/papers/{paper_id}",
            "published": str(paper.get("publishedAt") or item.get("publishedAt") or "")[:10],
            "title": _compact(paper.get("title") or item.get("title")),
            "summary": _compact(summary),
            "upvotes": _as_int(paper.get("upvotes") or item.get("upvotes")),
            "github": paper.get("githubRepo") or item.get("githubRepo") or "",
            "stars": _as_int(paper.get("githubStars") or item.get("githubStars")),
        })
    return records


# ---- TODO 4: web search / fetch through the Exa MCP endpoint ----
@tool
def web_search(query: str, objective: str = "", num_results: int = 5) -> str:
    """Search the web (Exa). Describe the ideal page in natural language. Returns clean text of the top results with URLs."""
    # PSEUDO-CODE:
    #   call the MCP tool "web_search_exa" with arguments {query, objective, numResults}
    #       (objective is REQUIRED by Exa: when empty, build one from the query)
    #   see GUIDE.md part 1.4 for how to call an MCP server over plain HTTP (JSON-RPC "tools/call") and read the answer
    #   read optional env EXA_API_KEY; when present it is sent to the Exa endpoint.
    #       (see GUIDE.md 1.4 for where it goes) => the key then appears in exception text: redact it before returning "ERROR: ..."
    #   WATCH OUT: read GUIDE.md 1.4 about how Exa signals "rate limited" on the free tier, and retry on it
    if not (query or "").strip():
        return "NO RESULTS"
    try:
        return _exa_call("web_search_exa", {
            "query": query.strip(),
            "objective": objective.strip() or f"Find reliable sources about {query.strip()}",
            "numResults": max(1, min(_as_int(num_results, 5), 10)),
        })
    except Exception as exc:
        return f"ERROR: {_error_text(exc, os.getenv('EXA_API_KEY', ''))}"


@tool
def web_fetch(url: str) -> str:
    """Read the full content of one web page (e.g. an arXiv abstract page) as markdown. Long pages are truncated."""
    # PSEUDO-CODE: MCP tool "web_fetch_exa" with arguments {"urls": [url]}; truncate the text to ~12000 chars
    if not (url or "").strip():
        return "NO RESULTS"
    try:
        return _exa_call("web_fetch_exa", {"urls": [url.strip()]})[:12000] or "NO RESULTS"
    except Exception as exc:
        return f"ERROR: {_error_text(exc, os.getenv('EXA_API_KEY', ''))}"


def _exa_call(name, arguments):
    secret = os.getenv("EXA_API_KEY", "")
    endpoint = EXA_URL + (f"?{urlencode({'exaApiKey': secret})}" if secret else "")
    payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": arguments}}

    def call():
        response = _request("post", endpoint, headers={
            "Content-Type": "application/json", "Accept": "application/json, text/event-stream",
        }, json=payload)
        messages = []
        for line in response.text.splitlines():
            if line.startswith("data:"):
                messages.append(json.loads(line[5:].strip()))
        answer = messages[-1] if messages else response.json()
        if answer.get("error"):
            raise ValueError(answer["error"])
        result = answer.get("result", {})
        meta = json.dumps(result.get("_meta", {})).lower()
        text = "\n".join(part.get("text", "") for part in result.get("content", []) if part.get("type") == "text")
        if "rate" in meta and ("limit" in meta or "quota" in meta):
            raise RetryableError("Exa rate limited")
        return text

    try:
        return with_retry(call, attempts=6, cap=60.0)
    except Exception as exc:
        # Preserve exception type but ensure a URL-bearing httpx error cannot expose the key.
        if secret:
            raise RuntimeError(_error_text(exc, secret)) from None
        raise


# ---- TODO 5: registry (the researcher subagent gets exactly these) ----
SOURCE_TOOLS = [arxiv_search, hf_daily_papers, hf_search_papers, web_search, web_fetch]


if __name__ == "__main__":
    for name, fn, args in [
        ("arxiv_search", arxiv_search, {"query": "world model", "max_results": 3}),
        ("hf_daily_papers", hf_daily_papers, {"limit": 20}),
        ("hf_search_papers", hf_search_papers, {"query": "world model", "limit": 3}),
        ("web_search", web_search, {"query": "survey paper on world models", "num_results": 2}),
        ("web_fetch", web_fetch, {"url": "https://arxiv.org/abs/1803.10122"}),
    ]:
        try:
            print(f"== {name}\n{fn.invoke(args)[:400]}\n")
        except NotImplementedError as exc:
            print(f"== {name}: not implemented yet ({exc})\n")
