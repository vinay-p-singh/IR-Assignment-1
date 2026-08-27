"""Term dictionary: the sorted vocabulary and its document-frequency column."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Vocabulary:
    """Sorted term list plus df, kept separate from the postings themselves.

    Real systems split these because the dictionary is small enough to hold in
    memory while the postings are not. Here the split is for clarity, and so the
    report can quote dictionary statistics independently of index statistics.
    """

    terms: tuple[str, ...]              # sorted, unique
    document_frequency: dict[str, int]
    collection_frequency: dict[str, int]

    def __len__(self) -> int:
        return len(self.terms)

    def __contains__(self, term: str) -> bool:
        return term in self.document_frequency

    def df(self, term: str) -> int:
        return self.document_frequency.get(term, 0)

    def cf(self, term: str) -> int:
        return self.collection_frequency.get(term, 0)

    def most_frequent(self, n: int = 20) -> list[tuple[str, int]]:
        return sorted(
            self.collection_frequency.items(), key=lambda kv: (-kv[1], kv[0])
        )[:n]

    def hapax_count(self) -> int:
        """Terms occurring exactly once. A blunt proxy for typos and names."""
        return sum(1 for c in self.collection_frequency.values() if c == 1)

    def stats(self) -> dict:
        return {
            "vocabulary_size": len(self.terms),
            "total_postings": sum(self.document_frequency.values()),
            "total_tokens": sum(self.collection_frequency.values()),
            "hapax_legomena": self.hapax_count(),
            "mean_df": round(
                sum(self.document_frequency.values()) / len(self.terms), 3
            )
            if self.terms
            else 0.0,
        }
