"""Deterministic address-defect scoring for Indian shipping addresses.

No LLM, no embeddings: a courier cannot deliver to "Near Hanuman Temple,
Ward 4" no matter how semantically rich it is, and a merchant cannot
audit a decision that came out of a black box. We score the *structural*
completeness a last-mile rider actually needs: a house/flat identifier, a
street/locality anchor, consistency with the PIN, and absence of junk.

Returns a defect score in [0, 1] plus the individual boolean signals so
the feature vector and the reason codes share one source of truth.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict

from ..data import pincodes

_HOUSE_RE = re.compile(
    r"(?:\b(?:h\.?\s?no|house\s?no|flat|plot|door|shop|unit|villa|apt|apartment|bldg|building|"
    r"floor|blk|block)\b[\s.:#-]*[a-z]?-?\d+[a-z/-]*\d*)|(?:^\s*#?\s*\d{1,5}[a-z/-]?\b)|"
    r"(?:\b\d{1,4}\s*[/-]\s*\d{1,4}\b)",
    re.I,
)
_STREET_RE = re.compile(
    r"\b(?:road|rd|street|st|lane|marg|nagar|colony|sector|layout|society|enclave|vihar|puram|"
    r"cross|main|phase|extension|extn|gali|chowk|bazaar|market|mohalla|tola|para|pally|nagar|"
    r"gram|village|town|taluk|tehsil|mandal|block|ward|post|po|ps)\b",
    re.I,
)
_LANDMARK_RE = re.compile(r"\b(?:near|nr|opp|opposite|behind|beside|next\s+to|in\s+front\s+of|b/h)\b", re.I)
_VAGUE_ONLY_RE = re.compile(
    r"\b(?:village|gaon|gram|ward|po|post\s*office|temple|mandir|masjid|church|school|bus\s*stand|"
    r"station|hospital|chowk|main\s*road)\b",
    re.I,
)
_JUNK_RE = re.compile(r"(?:(.)\1{3,})|\b(?:test|asdf|qwer|xyz|abc|na|n/a|nil|none|null|dummy|xxx)\b", re.I)
_PIN_IN_TEXT_RE = re.compile(r"\b[1-9]\d{5}\b")

# a few state/city tokens for PIN consistency (lower-cased)
_STATE_TOKENS = {
    "delhi": "Delhi", "mumbai": "Maharashtra", "thane": "Maharashtra", "pune": "Maharashtra", "nagpur": "Maharashtra",
    "maharashtra": "Maharashtra", "bengaluru": "Karnataka", "bangalore": "Karnataka", "karnataka": "Karnataka",
    "chennai": "Tamil Nadu", "tamil nadu": "Tamil Nadu", "tamilnadu": "Tamil Nadu", "hyderabad": "Telangana",
    "telangana": "Telangana", "kolkata": "West Bengal", "west bengal": "West Bengal", "bengal": "West Bengal",
    "ahmedabad": "Gujarat", "surat": "Gujarat", "gujarat": "Gujarat", "jaipur": "Rajasthan", "rajasthan": "Rajasthan",
    "lucknow": "Uttar Pradesh", "kanpur": "Uttar Pradesh", "uttar pradesh": "Uttar Pradesh", "up": "Uttar Pradesh",
    "patna": "Bihar", "bihar": "Bihar", "kerala": "Kerala", "kochi": "Kerala", "punjab": "Punjab",
    "haryana": "Haryana", "gurgaon": "Haryana", "gurugram": "Haryana", "noida": "Uttar Pradesh",
    "odisha": "Odisha", "bhubaneswar": "Odisha", "assam": "Assam", "guwahati": "Assam", "jharkhand": "Jharkhand",
    "ranchi": "Jharkhand", "madhya pradesh": "Madhya Pradesh", "indore": "Madhya Pradesh", "bhopal": "Madhya Pradesh",
    "chhattisgarh": "Chhattisgarh", "raipur": "Chhattisgarh", "andhra": "Andhra Pradesh", "vizag": "Andhra Pradesh",
    "visakhapatnam": "Andhra Pradesh",
}


@dataclass(frozen=True)
class AddressSignals:
    tokens: int
    has_house_no: bool
    has_street_anchor: bool
    landmark_only: bool
    vague_only: bool
    has_junk: bool
    pin_in_text_mismatch: bool
    state_mismatch: bool
    too_short: bool
    defect_score: float

    def as_dict(self) -> dict:
        return asdict(self)


def score_address(address: str, pin: str | int | None) -> AddressSignals:
    text = (address or "").strip()
    low = text.lower()
    tokens = [t for t in re.split(r"[\s,]+", low) if t]
    n = len(tokens)

    has_house = bool(_HOUSE_RE.search(text))
    has_street = bool(_STREET_RE.search(text))
    has_landmark = bool(_LANDMARK_RE.search(text))
    landmark_only = has_landmark and not has_house
    vague_only = (not has_house) and bool(_VAGUE_ONLY_RE.search(text)) and n <= 8
    junk = bool(_JUNK_RE.search(text)) or n == 0
    too_short = n < 4

    pin_info = pincodes.lookup(pin)
    pins_in_text = _PIN_IN_TEXT_RE.findall(text)
    pin_text_mismatch = any(p != pin_info.pin for p in pins_in_text) if pins_in_text else False

    state_mismatch = False
    for tok, st in _STATE_TOKENS.items():
        if re.search(rf"\b{re.escape(tok)}\b", low):
            if pin_info.valid and st != pin_info.state:
                state_mismatch = True
                break

    # Weighted defect score. Weights reflect last-mile failure lift observed in
    # Indian 3PL NDR (non-delivery report) data: a missing house number is the
    # single strongest predictor of "address not found".
    score = 0.0
    if not has_house:      score += 0.35
    if not has_street:     score += 0.15
    if landmark_only:      score += 0.15
    if vague_only:         score += 0.15
    if too_short:          score += 0.15
    if junk:               score += 0.35
    if pin_text_mismatch:  score += 0.20
    if state_mismatch:     score += 0.25
    if not pin_info.valid: score += 0.30
    score = min(1.0, score)

    return AddressSignals(
        tokens=n, has_house_no=has_house, has_street_anchor=has_street,
        landmark_only=landmark_only, vague_only=vague_only, has_junk=junk,
        pin_in_text_mismatch=pin_text_mismatch, state_mismatch=state_mismatch,
        too_short=too_short, defect_score=round(score, 3),
    )
