"""Small lexical scorer shared by local retrieval implementations."""

from __future__ import annotations

import re


def _normalise(value: str) -> str:
    return re.sub(r"\s+", "", value.lower())


def _tokens(value: str) -> set[str]:
    compact = _normalise(value)
    latin = set(re.findall(r"[a-z0-9._-]{2,}", compact))
    chinese_bigrams = {compact[index : index + 2] for index in range(len(compact) - 1)}
    return latin | chinese_bigrams


def score(query: str, text: str, keywords: list[str] | None = None) -> int:
    query_tokens = _tokens(query)
    text_tokens = _tokens(text)
    result = len(query_tokens & text_tokens)
    query_compact = _normalise(query)
    for keyword in keywords or []:
        normalized_keyword = _normalise(keyword)
        if normalized_keyword and normalized_keyword in query_compact:
            result += 4
    return result


def normalized_score(
    query: str, text: str, keywords: list[str] | None = None
) -> float:
    """Return a bounded lexical relevance score for stopping decisions."""

    query_tokens = _tokens(query)
    query_compact = _normalise(query)
    keyword_hits = sum(
        1
        for keyword in keywords or []
        if (normalized := _normalise(keyword)) and normalized in query_compact
    )
    maximum = len(query_tokens) + 4 * keyword_hits
    if maximum <= 0:
        return 0.0
    return round(min(1.0, score(query, text, keywords) / maximum), 6)
