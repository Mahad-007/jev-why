"""Splitting a state into independently maskable spans.

Pure: no network, no filesystem.

A Chunker does not return spans; it returns a Segmentation, which is a
reconstruction function. That is the load-bearing design decision in this
module. Masking logic belongs with the structure that knows how to rebuild
itself, not in a separate masker that has to re-parse the state and guess at
offsets.

Every Segmentation must satisfy `render(all_indices) == original`. If a chunker
cannot round-trip, its attributions are partly measuring its own lossiness, so
that property is tested for every implementation.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from collections.abc import Set as AbstractSet
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from jev_why.types import MaskMode, Span, SpanKind, State

TEXT_PLACEHOLDER = "[…]"
"""A typographic elision marker. Short, low-surprisal, and common in real prose,
which matters because the placeholder itself is a signal the model can react
to."""

JSON_STRING_PLACEHOLDER = "[…]"

CHARS_PER_TOKEN = 3.6
"""Rough conversion for budget pre-flight. It is only a starting point: the
budget module self-calibrates against the input_tokens every response reports."""


def estimate_tokens(text: str) -> int:
    return max(1, round(len(text) / CHARS_PER_TOKEN))


@runtime_checkable
class Segmentation(Protocol):
    @property
    def spans(self) -> Sequence[Span]: ...

    @property
    def mask_mode(self) -> MaskMode: ...

    def render(self, keep: AbstractSet[int]) -> State: ...


class Chunker(Protocol):
    def segment(self, state: State) -> Segmentation: ...


# --------------------------------------------------------------------------
# Text
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TextSegmentation:
    """Spans are half-open character ranges into `original`.

    Whatever sits between spans is glue -- whitespace, punctuation, markup --
    and is never masked. Glue carries structure rather than content, so
    removing it would perturb the document in a way that has nothing to do with
    the span under test.
    """

    original: str
    _spans: tuple[Span, ...]
    _mask_mode: MaskMode = MaskMode.REDACT
    placeholder: str = TEXT_PLACEHOLDER

    @property
    def spans(self) -> Sequence[Span]:
        return self._spans

    @property
    def mask_mode(self) -> MaskMode:
        return self._mask_mode

    def render(self, keep: AbstractSet[int]) -> str:
        out: list[str] = []
        cursor = 0
        for span in self._spans:
            assert span.start is not None and span.end is not None
            out.append(self.original[cursor : span.start])
            if span.index in keep:
                out.append(self.original[span.start : span.end])
            elif self._mask_mode is MaskMode.REDACT:
                out.append(self.placeholder)
            cursor = span.end
        out.append(self.original[cursor:])
        return "".join(out)


_ABBREVIATIONS = frozenset(
    [
        "mr",
        "mrs",
        "ms",
        "dr",
        "prof",
        "sr",
        "jr",
        "st",
        "vs",
        "etc",
        "e.g",
        "i.e",
        "cf",
        "al",
        "inc",
        "ltd",
        "co",
        "corp",
        "dept",
        "fig",
        "no",
        "vol",
        "pp",
        "ca",
        "approx",
        "est",
        "min",
        "max",
        "sec",
        "hr",
        "hrs",
        "am",
        "pm",
        "u.s",
        "u.k",
        "e.u",
    ]
)

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])[\"')\]]*\s+")


def _looks_like_abbreviation(text: str, end: int) -> bool:
    """True when the period at `end-1` terminates a known abbreviation."""
    tail = text[max(0, end - 12) : end].rstrip("\"')]")
    if not tail.endswith("."):
        return False
    word = re.split(r"[\s(]", tail[:-1])[-1].lower()
    if word in _ABBREVIATIONS:
        return True
    # A single initial such as "J." in "J. Smith" is not a sentence end.
    return len(word) == 1 and word.isalpha()


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Character ranges of sentences, excluding the whitespace that separates
    them. Ranges are returned in document order and never overlap."""
    spans: list[tuple[int, int]] = []
    start = 0
    for match in _SENTENCE_BOUNDARY.finditer(text):
        end = match.start()
        if end <= start or _looks_like_abbreviation(text, end):
            continue
        spans.append((start, end))
        start = match.end()
    if start < len(text):
        spans.append((start, len(text)))
    return [(s, e) for s, e in spans if text[s:e].strip()]


def split_blocks(text: str, *, granularity: str = "paragraph") -> list[tuple[int, int]]:
    """Character ranges of paragraphs (blank-line delimited) or raw lines."""
    pattern = r"\n[ \t]*\n" if granularity == "paragraph" else r"\n"
    spans: list[tuple[int, int]] = []
    start = 0
    for match in re.finditer(pattern, text):
        spans.append((start, match.start()))
        start = match.end()
    spans.append((start, len(text)))
    return [(s, e) for s, e in spans if text[s:e].strip()]


def _merge_runts(
    text: str, ranges: list[tuple[int, int]], *, min_chars: int
) -> list[tuple[int, int]]:
    """Fold fragments shorter than `min_chars` into the previous span.

    A three-word fragment rarely carries an independent reason, and every span
    costs a call, so spending one on "Thanks!" is waste.
    """
    if not ranges:
        return ranges
    merged: list[tuple[int, int]] = [ranges[0]]
    for start, end in ranges[1:]:
        prev_start, prev_end = merged[-1]
        if end - start < min_chars or prev_end - prev_start < min_chars:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    return merged


def _cap_spans(ranges: list[tuple[int, int]], *, max_spans: int) -> list[tuple[int, int]]:
    """Merge neighbours until at most `max_spans` remain.

    Cost grows with the number of spans AND with document size, so an uncapped
    chunker on a long document is quadratic. Capping is on by default.
    """
    if max_spans <= 0 or len(ranges) <= max_spans:
        return ranges
    group = -(-len(ranges) // max_spans)  # ceiling division
    out: list[tuple[int, int]] = []
    for i in range(0, len(ranges), group):
        chunk = ranges[i : i + group]
        out.append((chunk[0][0], chunk[-1][1]))
    return out


def _text_spans(
    text: str, ranges: list[tuple[int, int]], kind: SpanKind, prefix: str
) -> tuple[Span, ...]:
    return tuple(
        Span(
            index=i,
            label=f"{prefix}{i:03d}",
            text=text[s:e],
            kind=kind,
            start=s,
            end=e,
            tokens=estimate_tokens(text[s:e]),
        )
        for i, (s, e) in enumerate(ranges)
    )


@dataclass(frozen=True)
class SentenceChunker:
    min_chars: int = 40
    max_spans: int = 64
    mask_mode: MaskMode = MaskMode.REDACT

    def segment(self, state: State) -> TextSegmentation:
        text = _require_text(state, "SentenceChunker")
        ranges = _cap_spans(
            _merge_runts(text, split_sentences(text), min_chars=self.min_chars),
            max_spans=self.max_spans,
        )
        return TextSegmentation(
            text, _text_spans(text, ranges, SpanKind.SENTENCE, "s"), self.mask_mode
        )


@dataclass(frozen=True)
class BlockChunker:
    granularity: str = "paragraph"
    min_chars: int = 0
    max_spans: int = 64
    mask_mode: MaskMode = MaskMode.REDACT

    def segment(self, state: State) -> TextSegmentation:
        text = _require_text(state, "BlockChunker")
        ranges = _cap_spans(
            _merge_runts(
                text, split_blocks(text, granularity=self.granularity), min_chars=self.min_chars
            ),
            max_spans=self.max_spans,
        )
        kind = SpanKind.PARAGRAPH if self.granularity == "paragraph" else SpanKind.LINE
        prefix = "p" if self.granularity == "paragraph" else "l"
        return TextSegmentation(text, _text_spans(text, ranges, kind, prefix), self.mask_mode)


# --------------------------------------------------------------------------
# JSON
# --------------------------------------------------------------------------


def _walk_leaves(node: Any, path: str = "$") -> Iterable[tuple[str, Any]]:
    if isinstance(node, dict):
        for key in node:
            yield from _walk_leaves(node[key], f"{path}.{key}")
    elif isinstance(node, list):
        for i, item in enumerate(node):
            yield from _walk_leaves(item, f"{path}[{i}]")
    else:
        yield path, node


def _set_at(root: Any, path: str, value: Any) -> None:
    tokens = re.findall(r"\.([^.\[\]]+)|\[(\d+)\]", path)
    node = root
    for i, (key, idx) in enumerate(tokens):
        last = i == len(tokens) - 1
        step: Any = int(idx) if idx else key
        if last:
            node[step] = value
        else:
            node = node[step]


@dataclass(frozen=True)
class JsonSegmentation:
    """Spans are scalar leaves addressed by path.

    Masked leaves become `null` rather than being deleted. Deleting a key is a
    semantically different state -- "field not provided" is a signal a schema
    may treat as meaningful -- and deleting an array element renumbers every
    element after it, which would make the value of a coalition depend on more
    than the coalition. That last point is disqualifying: it silently breaks the
    additivity every estimator here rests on.
    """

    original: dict[str, Any] | list[Any]
    _spans: tuple[Span, ...]
    _paths: tuple[str, ...]
    _mask_mode: MaskMode = MaskMode.REDACT
    null_policy: str = "null"

    @property
    def spans(self) -> Sequence[Span]:
        return self._spans

    @property
    def mask_mode(self) -> MaskMode:
        return self._mask_mode

    def render(self, keep: AbstractSet[int]) -> dict[str, Any] | list[Any]:
        out = deepcopy(self.original)
        replacement: Any = None if self.null_policy == "null" else JSON_STRING_PLACEHOLDER
        for span, path in zip(self._spans, self._paths, strict=True):
            if span.index not in keep:
                _set_at(out, path, replacement)
        return out


@dataclass(frozen=True)
class JsonLeafChunker:
    max_spans: int = 96
    exclude: tuple[str, ...] = ()
    """Leaf-name substrings to leave unmasked. Masking a UUID or a timestamp
    spends a call to learn nothing."""

    null_policy: str = "null"
    mask_mode: MaskMode = MaskMode.REDACT

    def segment(self, state: State) -> JsonSegmentation:
        if not isinstance(state, (dict, list)):
            raise TypeError("JsonLeafChunker needs a dict or list state")
        leaves = [
            (path, value)
            for path, value in _walk_leaves(state)
            if not any(token in path for token in self.exclude)
        ]
        if self.max_spans > 0:
            leaves = leaves[: self.max_spans]
        spans = tuple(
            Span(
                index=i,
                label=path,
                text=str(value),
                kind=SpanKind.JSON_LEAF,
                tokens=estimate_tokens(str(value)),
            )
            for i, (path, value) in enumerate(leaves)
        )
        return JsonSegmentation(
            state, spans, tuple(p for p, _ in leaves), self.mask_mode, self.null_policy
        )


def _require_text(state: State, who: str) -> str:
    if not isinstance(state, str):
        raise TypeError(f"{who} needs a string state, got {type(state).__name__}")
    return state


def auto_chunker(state: State) -> Chunker:
    """Pick a reasonable chunker for the shape of the state."""
    if isinstance(state, (dict, list)):
        return JsonLeafChunker()
    if "\n\n" in state and len(state) > 2000:
        return BlockChunker()
    return SentenceChunker()


def resolve_chunker(chunker: Chunker | str, state: State) -> Chunker:
    if not isinstance(chunker, str):
        return chunker
    match chunker:
        case "auto":
            return auto_chunker(state)
        case "sentences":
            return SentenceChunker()
        case "paragraphs":
            return BlockChunker(granularity="paragraph")
        case "lines":
            return BlockChunker(granularity="line")
        case "json":
            return JsonLeafChunker()
        case _:
            raise ValueError(f"unknown chunker {chunker!r}")


__all__ = [
    "BlockChunker",
    "Chunker",
    "JsonLeafChunker",
    "JsonSegmentation",
    "Segmentation",
    "SentenceChunker",
    "TextSegmentation",
    "auto_chunker",
    "estimate_tokens",
    "resolve_chunker",
    "split_blocks",
    "split_sentences",
]
