"""Compute ROUGE-1 overlap between reference and generated summaries."""

from collections import Counter


def rouge_1_score(reference: str, candidate: str) -> dict:
    ref_tokens = reference.lower().split()
    cand_tokens = candidate.lower().split()
    ref_counter = Counter(ref_tokens)
    cand_counter = Counter(cand_tokens)
    overlap = sum((min(ref_counter[w], cand_counter[w]) for w in cand_counter))
    precision = overlap / len(cand_tokens) if cand_tokens else 0.0
    recall = overlap / len(ref_tokens) if ref_tokens else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}
