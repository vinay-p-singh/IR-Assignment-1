"""Boolean query parsing and evaluation over our own postings lists.

Two execution strategies are provided so the report can compare them:

    naive      -- evaluate AND arguments left to right, as written
    optimized  -- reorder AND arguments by increasing postings length

Both must return identical result sets. That equality is the correctness gate;
the comparison counts and timings are only meaningful once it holds.
"""

import re
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable

from .index import InvertedIndex

# ---------------------------------------------------------------- query AST


@dataclass(frozen=True)
class Term:
    value: str


@dataclass(frozen=True)
class Not:
    child: "Node"


@dataclass(frozen=True)
class And:
    children: tuple["Node", ...]


@dataclass(frozen=True)
class Or:
    children: tuple["Node", ...]


Node = Term | Not | And | Or

# ---------------------------------------------------------------- parsing

_QUERY_TOKEN_RE = re.compile(r"\(|\)|[A-Za-z0-9]+(?:['\-][A-Za-z0-9]+)*")
_OPERATORS = {"AND", "OR", "NOT"}


class QuerySyntaxError(ValueError):
    pass


def tokenize_query(query: str) -> list[str]:
    return _QUERY_TOKEN_RE.findall(query)


class _Parser:
    """Recursive descent. Precedence: NOT binds tighter than AND, then OR.

    Adjacent operands with no operator between them are treated as AND, so
    'oil price' and 'oil AND price' mean the same thing.
    """

    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> str | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def next(self) -> str:
        tok = self.peek()
        if tok is None:
            raise QuerySyntaxError("unexpected end of query")
        self.pos += 1
        return tok

    def parse(self) -> Node:
        if not self.tokens:
            raise QuerySyntaxError("empty query")
        node = self.parse_or()
        if self.peek() is not None:
            raise QuerySyntaxError(f"unexpected token {self.peek()!r}")
        return node

    def parse_or(self) -> Node:
        children = [self.parse_and()]
        while self.peek() and self.peek().upper() == "OR":
            self.next()
            children.append(self.parse_and())
        return children[0] if len(children) == 1 else Or(tuple(children))

    def parse_and(self) -> Node:
        children = [self.parse_not()]
        while True:
            tok = self.peek()
            if tok is None or tok == ")" or tok.upper() == "OR":
                break
            if tok.upper() == "AND":
                self.next()
            children.append(self.parse_not())
        return children[0] if len(children) == 1 else And(tuple(children))

    def parse_not(self) -> Node:
        tok = self.peek()
        if tok and tok.upper() == "NOT":
            self.next()
            return Not(self.parse_not())
        return self.parse_atom()

    def parse_atom(self) -> Node:
        tok = self.next()
        if tok == "(":
            node = self.parse_or()
            closing = self.next()
            if closing != ")":
                raise QuerySyntaxError("missing closing parenthesis")
            return node
        if tok == ")":
            raise QuerySyntaxError("unbalanced closing parenthesis")
        if tok.upper() in _OPERATORS:
            raise QuerySyntaxError(f"operator {tok!r} used as a term")
        return Term(tok)


def parse(query: str) -> Node:
    return _Parser(tokenize_query(query)).parse()


# ---------------------------------------------------------------- merges


@dataclass
class QueryStats:
    comparisons: int = 0
    elapsed_ms: float = 0.0
    postings_touched: int = 0
    order: list[str] = field(default_factory=list)


def intersect(a: list[int], b: list[int], stats: QueryStats) -> list[int]:
    out: list[int] = []
    i = j = 0
    while i < len(a) and j < len(b):
        stats.comparisons += 1
        if a[i] == b[j]:
            out.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return out


def union(a: list[int], b: list[int], stats: QueryStats) -> list[int]:
    out: list[int] = []
    i = j = 0
    while i < len(a) and j < len(b):
        stats.comparisons += 1
        if a[i] == b[j]:
            out.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            out.append(a[i])
            i += 1
        else:
            out.append(b[j])
            j += 1
    out.extend(a[i:])
    out.extend(b[j:])
    return out


def difference(a: list[int], b: list[int], stats: QueryStats) -> list[int]:
    """a AND NOT b, without ever materialising the complement of b."""
    out: list[int] = []
    i = j = 0
    while i < len(a) and j < len(b):
        stats.comparisons += 1
        if a[i] == b[j]:
            i += 1
            j += 1
        elif a[i] < b[j]:
            out.append(a[i])
            i += 1
        else:
            j += 1
    out.extend(a[i:])
    return out


# ---------------------------------------------------------------- evaluation

TermNormalizer = Callable[[str], str]


def _normalized_postings(
    node: Term, index: InvertedIndex, normalize: TermNormalizer, stats: QueryStats
) -> list[int]:
    term = normalize(node.value)
    postings = index.postings(term)
    stats.postings_touched += len(postings)
    stats.order.append(f"{term}({len(postings)})")
    return postings


def _estimated_size(
    node: Node, index: InvertedIndex, normalize: TermNormalizer
) -> int:
    """Cheap size prediction used to order AND arguments."""
    match node:
        case Term(value):
            return index.df(normalize(value))
        case Not(child):
            return index.num_docs - _estimated_size(child, index, normalize)
        case And(children):
            return min(_estimated_size(c, index, normalize) for c in children)
        case Or(children):
            return min(
                index.num_docs,
                sum(_estimated_size(c, index, normalize) for c in children),
            )
    raise TypeError(f"unknown node {node!r}")


def _evaluate(
    node: Node,
    index: InvertedIndex,
    normalize: TermNormalizer,
    stats: QueryStats,
    optimize: bool,
) -> list[int]:
    match node:
        case Term():
            return _normalized_postings(node, index, normalize, stats)

        case Not(child):
            inner = _evaluate(child, index, normalize, stats, optimize)
            return difference(index.all_doc_ids, inner, stats)

        case Or(children):
            result = _evaluate(children[0], index, normalize, stats, optimize)
            for child in children[1:]:
                result = union(
                    result, _evaluate(child, index, normalize, stats, optimize), stats
                )
            return result

        case And(children):
            ordered = list(children)
            if optimize:
                # Smallest postings list first bounds the size of every
                # intermediate result, and NOT arguments are deferred so they
                # can be answered with difference() instead of a complement.
                ordered.sort(
                    key=lambda c: (
                        isinstance(c, Not),
                        _estimated_size(c, index, normalize),
                    )
                )
            result = _evaluate(ordered[0], index, normalize, stats, optimize)
            for child in ordered[1:]:
                if optimize and isinstance(child, Not):
                    excluded = _evaluate(
                        child.child, index, normalize, stats, optimize
                    )
                    result = difference(result, excluded, stats)
                else:
                    result = intersect(
                        result,
                        _evaluate(child, index, normalize, stats, optimize),
                        stats,
                    )
                if not result:
                    break
            return result

    raise TypeError(f"unknown node {node!r}")


def execute(
    query: str,
    index: InvertedIndex,
    normalize: TermNormalizer,
    optimize: bool = False,
) -> tuple[list[int], QueryStats]:
    node = parse(query)
    stats = QueryStats()
    start = time.perf_counter()
    result = _evaluate(node, index, normalize, stats, optimize)
    stats.elapsed_ms = (time.perf_counter() - start) * 1000
    return result, stats


def render(node: Node) -> str:
    """Readable form of a parsed query, for the report's query traces."""
    match node:
        case Term(value):
            return value
        case Not(child):
            return f"NOT {render(child)}"
        case And(children):
            return "(" + " AND ".join(render(c) for c in children) + ")"
        case Or(children):
            return "(" + " OR ".join(render(c) for c in children) + ")"
    raise TypeError(f"unknown node {node!r}")


def normalizer_for(preprocessor) -> TermNormalizer:
    """Query terms must pass through the same pipeline the index was built with."""

    def normalize(term: str) -> str:
        tokens = preprocessor(term)
        return tokens[0] if tokens else term.lower()

    return normalize


def all_terms(node: Node) -> Iterable[str]:
    match node:
        case Term(value):
            yield value
        case Not(child):
            yield from all_terms(child)
        case And(children) | Or(children):
            for c in children:
                yield from all_terms(c)
