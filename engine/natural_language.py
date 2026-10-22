from __future__ import annotations

import re


_RESULT_NOUNS = {
    "link", "links", "edge", "edges", "road", "roads", "location", "locations",
    "stage", "stages", "cell", "cells", "result", "results", "area", "areas",
    "component", "components", "bottleneck", "bottlenecks", "hotspot", "hotspots",
}
_REQUEST_VERBS = {
    "identify", "find", "show", "give", "return", "list", "display", "rank",
    "select", "report", "provide", "get", "need", "want",
}
_RANK_WORDS = {
    "top", "first", "most", "least", "worst", "best", "highest", "lowest",
    "critical", "important", "priority", "ranked", "vulnerable", "degraded",
}
_SMALL = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_NUMBER_WORDS = set(_SMALL) | set(_TENS) | {"hundred", "thousand", "million", "and"}


def _positive(value: str | int, default: int) -> int:
    try:
        parsed = int(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _english_number(words: list[str]) -> int | None:
    total = 0
    current = 0
    saw_number = False
    for word in words:
        if word == "and":
            continue
        if word in _SMALL:
            current += _SMALL[word]
            saw_number = True
        elif word in _TENS:
            current += _TENS[word]
            saw_number = True
        elif word == "hundred":
            current = max(1, current) * 100
            saw_number = True
        elif word == "thousand":
            total += max(1, current) * 1_000
            current = 0
            saw_number = True
        elif word == "million":
            total = (total + max(1, current)) * 1_000_000
            current = 0
            saw_number = True
        else:
            return None
    value = total + current
    return value if saw_number and value > 0 else None


def extract_requested_count(text: str, default: int = 10) -> int:
    """Read an explicit requested result count without a fixed product cap."""
    lowered = " ".join(str(text or "").casefold().split())
    if not lowered:
        return default

    noun = r"(?:links?|edges?|roads?|locations?|stages?|cells?|results?|areas?|components?|bottlenecks?|hotspots?)"
    digits = r"([1-9][0-9,]{0,11})"
    patterns = (
        rf"\b(?:change|set|update)\s+(?:the\s+)?(?:top\s*[0-9,]+\s*)?(?:to|as)\s*{digits}\b",
        rf"\b(?:make\s+it|instead\s+return|instead\s+show)\s*{digits}\b",
        rf"\b(?:show|return|list|display|give|provide)\s+(?:me\s+)?{digits}\s+(?:instead|rather)\b",
        rf"\b(?:top|first)\s*[-:]?\s*{digits}\b",
        rf"\b(?:identify|find|show|give|return|list|display|rank|select|report|provide|get|need|want)\s+(?:me\s+)?(?:the\s+)?{digits}\s+(?:[a-z][a-z-]*\s+){{0,6}}{noun}\b",
        rf"\b{digits}\s+(?:(?:most|least|top|worst|best|highest|lowest|critical|important|priority|ranked|vulnerable|degraded|persistent|persistently|structural|geographic)\s+){{1,6}}{noun}\b",
        rf"\b{digits}\s+{noun}\b",
        rf"(?:\u524d|\u8fd4\u56de|\u663e\u793a|\u627e\u51fa|\u8bc6\u522b)\s*{digits}\s*(?:\u6761|\u4e2a|\u9879)?",
        rf"{digits}\s*(?:\u6761|\u4e2a|\u9879)\s*(?:\u5173\u952e\u8fb9|\u5173\u952e\u8def\u6bb5|\u8fb9|\u8def\u6bb5|\u9053\u8def|\u94fe\u8def|\u4f4d\u7f6e|\u9636\u6bb5|\u70ed\u70b9|\u7ed3\u679c|\u533a\u57df|\u7ec4\u4ef6)",
    )
    for pattern in patterns:
        match = re.search(pattern, lowered, flags=re.IGNORECASE)
        if match:
            return _positive(match.group(1), default)

    tokens = re.findall(r"[a-z]+", lowered)
    for start, token in enumerate(tokens):
        if token not in _NUMBER_WORDS or token == "and":
            continue
        end = start
        while end < len(tokens) and tokens[end] in _NUMBER_WORDS:
            end += 1
        value = _english_number(tokens[start:end])
        if value is None:
            continue
        before = set(tokens[max(0, start - 6):start])
        after = set(tokens[end:min(len(tokens), end + 8)])
        if (
            before & {"top", "first"}
            or (before & _REQUEST_VERBS and after & _RESULT_NOUNS)
            or (after & _RESULT_NOUNS and after & _RANK_WORDS)
        ):
            return value
    return default
