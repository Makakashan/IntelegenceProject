from __future__ import annotations

import time
import re

import requests

from .config import DEFAULT_SEARCH_RESULTS, DEFAULT_SERPER_URL, SERPER_API_KEY_ENV
from .logging import log, write_jsonl
from .quality import is_blocked_source


def serper_search(
    query: str,
    api_key: str,
    num: int = DEFAULT_SEARCH_RESULTS,
    serper_url: str = DEFAULT_SERPER_URL,
) -> dict:
    if not api_key:
        raise ValueError(f"{SERPER_API_KEY_ENV} is required for Serper search.")

    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    payload = {"q": query, "num": num, "gl": "gb", "hl": "en"}

    for attempt in range(4):
        started_at = time.perf_counter()
        resp = requests.post(serper_url, headers=headers, json=payload, timeout=12)
        elapsed = round(time.perf_counter() - started_at, 3)
        if resp.status_code == 429:
            wait = 2**attempt
            log.debug("Serper 429 - retry in %ds (attempt %d)", wait, attempt + 1)
            time.sleep(wait)
            continue

        resp.raise_for_status()
        data = resp.json()
        write_jsonl(
            {
                "api": "serper",
                "query": query,
                "status": resp.status_code,
                "elapsed_s": elapsed,
                "hits": len(data.get("organic", [])),
            }
        )
        return data

    resp.raise_for_status()
    return {}


def _tokens(text: str) -> set[str]:
    stopwords = {"offshore", "wind", "farm", "project", "the", "of", "and"}
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if token not in stopwords and len(token) > 1}


def has_budget_context(text: str) -> bool:
    return bool(
        re.search(
            r"\b(total\s+)?(investment|budget|cost|capex|project\s+cost|project\s+value|financial\s+close|financing|fid)\b",
            text,
            re.IGNORECASE,
        )
    )


def project_name_matches(farm_name: str, text: str) -> bool:
    farm_tokens = _tokens(farm_name)
    if not farm_tokens:
        return False
    text_tokens = _tokens(text)
    required = min(2, len(farm_tokens))
    return len(farm_tokens & text_tokens) >= required


def get_candidate_hits(results: dict, farm_name: str) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    for item in results.get("organic", []):
        title = item.get("title", "")
        snippet = item.get("snippet", "")
        link = item.get("link", "")
        text = f"{title}. {snippet}".strip()
        if not snippet or is_blocked_source(link):
            continue
        if not project_name_matches(farm_name, text):
            continue
        if not has_budget_context(text):
            continue
        hits.append({"text": text, "source": link, "title": title})
    return hits


def get_snippets(results: dict, farm_name: str | None = None) -> tuple[list[str], list[str]]:
    snippets, sources = [], []
    if farm_name is not None:
        hits = get_candidate_hits(results, farm_name)
        return [hit["text"] for hit in hits], [hit["source"] for hit in hits]

    for item in results.get("organic", []):
        snippet = item.get("snippet", "")
        link = item.get("link", "")
        if snippet and not is_blocked_source(link):
            snippets.append(f"{item.get('title', '')}. {snippet}")
            sources.append(link)

    for key in ("answerBox", "knowledgeGraph"):
        box = results.get(key, {})
        for field in ("answer", "description", "snippet"):
            value = box.get(field, "")
            if value:
                snippets.insert(0, value)
    return snippets, sources


def build_search_queries(farm_name: str) -> tuple[str, str]:
    return (
        f'"{farm_name}" offshore wind farm total investment budget cost billion',
        f"{farm_name} wind farm total project cost financial close",
    )
