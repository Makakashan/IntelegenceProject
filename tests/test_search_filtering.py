from __future__ import annotations

from src.budget.search import get_candidate_hits


def test_candidate_hits_reject_blocked_sources_and_unrelated_projects() -> None:
    results = {
        "organic": [
            {
                "title": "Dogger Bank update",
                "snippet": "The project cost is £32bn according to a viral post.",
                "link": "https://facebook.com/example",
            },
            {
                "title": "Another port investment",
                "snippet": "A £3 billion investment was announced for unrelated infrastructure.",
                "link": "https://example.com/port",
            },
            {
                "title": "Dogger Bank A reaches financial close",
                "snippet": "Dogger Bank A offshore wind farm project cost is GBP 3 billion.",
                "link": "https://orsted.com/en/news/example",
            },
        ]
    }

    hits = get_candidate_hits(results, "Dogger Bank A")

    assert len(hits) == 1
    assert hits[0]["source"] == "https://orsted.com/en/news/example"
