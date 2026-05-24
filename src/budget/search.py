from __future__ import annotations

import time

import requests

from .config import DEFAULT_SEARCH_RESULTS, DEFAULT_SERPER_URL, SERPER_API_KEY_ENV
from .logging import log, write_jsonl


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


def get_snippets(results: dict) -> tuple[list[str], list[str]]:
    snippets, sources = [], []
    for item in results.get("organic", []):
        snippet = item.get("snippet", "")
        if snippet:
            snippets.append(f"{item.get('title', '')}. {snippet}")
            sources.append(item.get("link", ""))

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
