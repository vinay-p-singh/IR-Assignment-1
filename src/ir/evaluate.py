"""Set-based retrieval effectiveness: precision, recall and F1.

Boolean retrieval returns an unranked set, so these are the set-theoretic
definitions rather than the ranked-list ones. Averaging is reported both ways
because they answer different questions: macro treats every query as equally
important, micro treats every document decision as equally important, and a
system that does well on one can do badly on the other.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Scores:
    query_id: str
    retrieved: int
    relevant: int
    true_positives: int

    @property
    def precision(self) -> float:
        return self.true_positives / self.retrieved if self.retrieved else 0.0

    @property
    def recall(self) -> float:
        return self.true_positives / self.relevant if self.relevant else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def as_row(self) -> dict:
        return {
            "query_id": self.query_id,
            "retrieved": self.retrieved,
            "relevant": self.relevant,
            "true_positives": self.true_positives,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


def score(query_id: str, retrieved: set[int], relevant: set[int]) -> Scores:
    return Scores(
        query_id=query_id,
        retrieved=len(retrieved),
        relevant=len(relevant),
        true_positives=len(retrieved & relevant),
    )


def macro_average(scores: list[Scores]) -> dict:
    """Mean of the per-query scores. Every query counts once."""
    if not scores:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    n = len(scores)
    return {
        "precision": round(sum(s.precision for s in scores) / n, 4),
        "recall": round(sum(s.recall for s in scores) / n, 4),
        "f1": round(sum(s.f1 for s in scores) / n, 4),
    }


def micro_average(scores: list[Scores]) -> dict:
    """Pool the counts first, then divide. Large result sets dominate."""
    tp = sum(s.true_positives for s in scores)
    retrieved = sum(s.retrieved for s in scores)
    relevant = sum(s.relevant for s in scores)
    precision = tp / retrieved if retrieved else 0.0
    recall = tp / relevant if relevant else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def summarise(scores: list[Scores]) -> dict:
    return {
        "num_queries": len(scores),
        "macro": macro_average(scores),
        "micro": micro_average(scores),
    }
