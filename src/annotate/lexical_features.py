"""Lexical feature extraction for uncertainty markers.

Extracts hedge words, uncertainty statements, self-corrections, and other
markers from reasoning artifacts and final answers. All features are computed
per-text-segment (artifact or answer) and returned as a structured dict.
"""

import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Marker lexicons
# ---------------------------------------------------------------------------

HEDGE_WORDS = [
    r"\blikely\b", r"\bprobably\b", r"\bmight\b", r"\bperhaps\b",
    r"\bpossibly\b", r"\bcould be\b", r"\bmay be\b", r"\bseems\b",
    r"\bappears\b", r"\bsuggest(?:s|ed)?\b", r"\btend(?:s)?\b",
    r"\bapproximately\b", r"\broughly\b", r"\bargua?bly\b",
]

UNCERTAINTY_STATEMENTS = [
    r"\bi(?:'m| am) not (?:entirely |completely |fully )?sure\b",
    r"\bi(?:'m| am) not (?:entirely |completely )?certain\b",
    r"\bi don(?:'t| not) know\b",
    r"\bit(?:'s| is) unclear\b",
    r"\bit(?:'s| is) uncertain\b",
    r"\bi(?:'m| am) uncertain\b",
    r"\bi(?:'m| am) unsure\b",
    r"\bhard to (?:say|tell|determine|know)\b",
    r"\bdifficult to (?:say|tell|determine|know)\b",
    r"\bnot enough information\b",
    r"\bcannot (?:be )?(?:determined|confirmed|verified)\b",
]

SELF_CORRECTIONS = [
    r"\bwait\b",
    r"\bactually\b",
    r"\blet me reconsider\b",
    r"\bon second thought\b",
    r"\bi (?:made a |was )(?:mistake|wrong|error)\b",
    r"\bcorrection\b",
    r"\bno,\s",
    r"\bhmm\b",
    r"\bhold on\b",
    r"\blet me (?:re)?think\b",
    r"\blet me re-?(?:examine|evaluate|check)\b",
    r"\bthat(?:'s| is) (?:not right|wrong|incorrect)\b",
]

ALTERNATIVE_HYPOTHESES = [
    r"\balternatively\b",
    r"\banother (?:possibility|explanation|interpretation|approach)\b",
    r"\bor it could be\b",
    r"\bon the other hand\b",
    r"\bit(?:'s| is) also possible\b",
    r"\bthere(?:'s| is) also (?:a |the )?(?:possibility|chance)\b",
]

DEFEATERS = [
    r"\bunless\b",
    r"\bhowever\b",
    r"\bbut if\b",
    r"\bexcept (?:that|when|if)\b",
    r"\bprovided that\b",
    r"\bassuming\b",
    r"\bcontingent on\b",
]

MISSING_INFO_REQUESTS = [
    r"\bi would need\b",
    r"\bnot enough information\b",
    r"\bit depends on\b",
    r"\bwithout (?:more |additional |further )?(?:information|context|data|details)\b",
    r"\binsufficient (?:information|data|context)\b",
    r"\bmore (?:information|context|data|details) (?:is |are |would be )?needed\b",
    r"\bcannot (?:be )?answered (?:without|with the given)\b",
]

ABSTENTION_MARKERS = [
    r"\bi cannot answer\b",
    r"\bi(?:'m| am) unable to (?:answer|provide|determine)\b",
    r"\bunanswerable\b",
    r"\bthis question cannot be answered\b",
    r"\bthere is no (?:definitive |clear )?answer\b",
    r"\bi (?:must |have to |need to )?abstain\b",
    r"\bi (?:don't|do not) have enough\b",
]


# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

def _compile_patterns(patterns: list[str]) -> list[re.Pattern]:
    return [re.compile(p, re.IGNORECASE) for p in patterns]


COMPILED = {
    "hedge": _compile_patterns(HEDGE_WORDS),
    "uncertainty_statement": _compile_patterns(UNCERTAINTY_STATEMENTS),
    "self_correction": _compile_patterns(SELF_CORRECTIONS),
    "alternative_hypothesis": _compile_patterns(ALTERNATIVE_HYPOTHESES),
    "defeater": _compile_patterns(DEFEATERS),
    "missing_info": _compile_patterns(MISSING_INFO_REQUESTS),
    "abstention": _compile_patterns(ABSTENTION_MARKERS),
}


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@dataclass
class LexicalFeatures:
    """Extracted lexical uncertainty features for a single text segment."""
    hedge_count: int = 0
    uncertainty_statement_count: int = 0
    self_correction_count: int = 0
    alternative_hypothesis_count: int = 0
    defeater_count: int = 0
    missing_info_count: int = 0
    abstention_count: int = 0
    total_markers: int = 0
    uncertainty_rate: float = 0.0  # markers per 100 tokens
    first_uncertainty_position: float = -1.0  # normalized 0.0-1.0, -1 if none
    has_revision_event: bool = False
    token_count: int = 0
    marker_positions: list[float] = field(default_factory=list)


def _count_matches(text: str, patterns: list[re.Pattern]) -> tuple[int, list[int]]:
    """Count total matches and return their character positions."""
    count = 0
    positions = []
    for pattern in patterns:
        for match in pattern.finditer(text):
            count += 1
            positions.append(match.start())
    return count, positions


def _estimate_token_count(text: str) -> int:
    """Rough token estimate: split on whitespace. Good enough for rate computation."""
    return len(text.split())


def extract_features(text: str) -> LexicalFeatures:
    """Extract lexical uncertainty features from a text segment."""
    if not text or not text.strip():
        return LexicalFeatures()

    token_count = _estimate_token_count(text)
    text_len = len(text)
    all_positions: list[int] = []

    counts = {}
    for category, patterns in COMPILED.items():
        count, positions = _count_matches(text, patterns)
        counts[category] = count
        all_positions.extend(positions)

    total_markers = sum(counts.values())

    # Compute first uncertainty position (normalized)
    first_pos = -1.0
    normalized_positions = []
    if all_positions and text_len > 0:
        all_positions.sort()
        first_pos = all_positions[0] / text_len
        normalized_positions = [p / text_len for p in all_positions]

    # Uncertainty rate: markers per 100 tokens
    rate = (total_markers / token_count * 100) if token_count > 0 else 0.0

    # Revision event: self-correction exists
    has_revision = counts["self_correction"] > 0

    return LexicalFeatures(
        hedge_count=counts["hedge"],
        uncertainty_statement_count=counts["uncertainty_statement"],
        self_correction_count=counts["self_correction"],
        alternative_hypothesis_count=counts["alternative_hypothesis"],
        defeater_count=counts["defeater"],
        missing_info_count=counts["missing_info"],
        abstention_count=counts["abstention"],
        total_markers=total_markers,
        uncertainty_rate=rate,
        first_uncertainty_position=first_pos,
        has_revision_event=has_revision,
        token_count=token_count,
        marker_positions=normalized_positions,
    )


def features_to_dict(features: LexicalFeatures) -> dict:
    """Convert LexicalFeatures to a flat dict for DataFrame construction."""
    return {
        "hedge_count": features.hedge_count,
        "uncertainty_statement_count": features.uncertainty_statement_count,
        "self_correction_count": features.self_correction_count,
        "alternative_hypothesis_count": features.alternative_hypothesis_count,
        "defeater_count": features.defeater_count,
        "missing_info_count": features.missing_info_count,
        "abstention_count": features.abstention_count,
        "total_markers": features.total_markers,
        "uncertainty_rate": features.uncertainty_rate,
        "first_uncertainty_position": features.first_uncertainty_position,
        "has_revision_event": features.has_revision_event,
        "token_count": features.token_count,
    }
