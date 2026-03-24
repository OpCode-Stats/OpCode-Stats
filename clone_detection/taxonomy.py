"""
Clone type classification (Type 1 / 2 / 3).
"""

import logging
from dataclasses import dataclass
from typing import Dict, List

from clone_detection.lsh import CandidatePair
from clone_detection.alignment import AlignmentResult

logger = logging.getLogger(__name__)

# Jaccard thresholds for clone types
THRESHOLDS: Dict[int, float] = {1: 0.95, 2: 0.70, 3: 0.50}


@dataclass
class ClonePair:
    func_id1: str
    func_id2: str
    jaccard: float
    alignment_score: float
    clone_type: int   # 1, 2, or 3
    binary1: str
    binary2: str
    cross_binary: bool


MIN_ALIGNMENT_SCORE = 0.30  # pairs below this are rejected regardless of Jaccard


def classify_clone_type(jaccard: float, alignment_score: float) -> int:
    """
    Classify a clone pair by type.

    Primary rule: Jaccard thresholds (Type 1 ≥ 0.95, Type 2 ≥ 0.70, Type 3 ≥ 0.50).
    Confirmation gate: pairs with alignment_score < MIN_ALIGNMENT_SCORE are
    rejected (return 0), ensuring Smith-Waterman acts as a genuine filter.
    Refinement: downgrade by one level if |jaccard - alignment_score| > 0.3,
    suggesting structural divergence despite high set similarity.
    """
    if jaccard < THRESHOLDS[3]:
        return 0  # not a clone

    # Smith-Waterman confirmation gate: reject if alignment score too low
    if alignment_score < MIN_ALIGNMENT_SCORE:
        return 0

    if jaccard >= THRESHOLDS[1]:
        t = 1
    elif jaccard >= THRESHOLDS[2]:
        t = 2
    else:
        t = 3

    # Downgrade if the two metrics disagree strongly
    if abs(jaccard - alignment_score) > 0.3:
        t = min(t + 1, 3)

    return t


def build_clone_pairs(
    candidates: List[CandidatePair],
    alignments: List[AlignmentResult],
) -> List[ClonePair]:
    """
    Merge candidate Jaccard scores with alignment scores and classify clone types.
    """
    # Index alignments by (func_id1, func_id2) canonical order
    align_map: Dict[tuple, AlignmentResult] = {}
    for ar in alignments:
        key = (min(ar.func_id1, ar.func_id2), max(ar.func_id1, ar.func_id2))
        align_map[key] = ar

    # Use best jaccard per pair across k values
    best_jaccard: Dict[tuple, float] = {}
    for cp in candidates:
        key = (min(cp.func_id1, cp.func_id2), max(cp.func_id1, cp.func_id2))
        if key not in best_jaccard or cp.jaccard_estimate > best_jaccard[key]:
            best_jaccard[key] = cp.jaccard_estimate

    clone_pairs: List[ClonePair] = []
    for key, jaccard in best_jaccard.items():
        ar = align_map.get(key)
        if ar is None:
            continue
        alignment_score = ar.normalized_score

        clone_type = classify_clone_type(jaccard, alignment_score)
        if clone_type == 0:
            continue

        fid1, fid2 = key
        binary1 = fid1.split('|')[0]
        binary2 = fid2.split('|')[0]

        clone_pairs.append(
            ClonePair(
                func_id1=fid1,
                func_id2=fid2,
                jaccard=jaccard,
                alignment_score=alignment_score,
                clone_type=clone_type,
                binary1=binary1,
                binary2=binary2,
                cross_binary=(binary1 != binary2),
            )
        )

    logger.info(
        f"Built {len(clone_pairs)} clone pairs from {len(best_jaccard)} unique candidates"
    )
    return clone_pairs
