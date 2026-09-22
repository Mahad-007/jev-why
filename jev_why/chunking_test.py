from __future__ import annotations

import pytest

from jev_why.chunking import (
    BlockChunker,
    JsonLeafChunker,
    SentenceChunker,
    auto_chunker,
    resolve_chunker,
    split_sentences,
)
from jev_why.types import MaskMode

TICKET = (
    "Hi, I was charged twice by Dr. Smith on the 3rd. Please fix this ASAP. "
    "I have been waiting for three days and nobody has replied. Thanks."
)


def _all(seg: object) -> frozenset[int]:
    return frozenset(range(len(seg.spans)))  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "chunker",
    [
        SentenceChunker(min_chars=10),
        BlockChunker(granularity="line"),
        BlockChunker(granularity="paragraph"),
    ],
)
def test_rendering_every_span_reproduces_the_original(chunker: object) -> None:
    """The round-trip property. A chunker that cannot rebuild its input is
    measuring its own lossiness alongside the model's behaviour."""
    seg = chunker.segment(TICKET)  # type: ignore[attr-defined]
    assert seg.render(_all(seg)) == TICKET


def test_json_chunker_round_trips() -> None:
    state = {"customer": {"tier": "gold"}, "items": [{"sku": "A"}, {"sku": "B"}]}
    seg = JsonLeafChunker().segment(state)
    assert seg.render(_all(seg)) == state


def test_abbreviations_do_not_end_a_sentence() -> None:
    assert len(split_sentences("Paid by Dr. Smith today.")) == 1
    assert len(split_sentences("Contact J. Smith now.")) == 1
    assert len(split_sentences("It failed. We retried.")) == 2


def test_redaction_preserves_surrounding_text() -> None:
    seg = SentenceChunker(min_chars=10).segment(TICKET)
    rendered = seg.render(_all(seg) - {1})
    assert "[…]" in rendered
    assert "charged twice" in rendered
    assert "waiting for three days" in rendered


def test_the_empty_coalition_is_a_redacted_state_not_an_empty_one() -> None:
    """v(0) is the null reference for every attribution. An empty string may be
    rejected outright, and would measure something other than absent content."""
    seg = SentenceChunker(min_chars=10).segment(TICKET)
    empty = seg.render(frozenset())
    assert empty.strip() != ""
    assert "charged" not in empty


def test_delete_mode_removes_rather_than_replaces() -> None:
    seg = SentenceChunker(min_chars=10, mask_mode=MaskMode.DELETE).segment(TICKET)
    rendered = seg.render(_all(seg) - {0})
    assert "[…]" not in rendered
    assert "charged twice" not in rendered


def test_json_leaves_become_null_so_array_indices_survive() -> None:
    """Deleting an array element renumbers everything after it, which would make
    a coalition's value depend on more than the coalition."""
    state = {"items": [{"sku": "A"}, {"sku": "B"}, {"sku": "C"}]}
    seg = JsonLeafChunker().segment(state)
    masked = seg.render(frozenset({0, 2}))
    assert masked == {"items": [{"sku": "A"}, {"sku": None}, {"sku": "C"}]}


def test_excluded_json_leaves_are_never_spans() -> None:
    state = {"id": "u-1", "note": "charged twice", "created_at": "2026-01-01"}
    seg = JsonLeafChunker(exclude=("id", "created_at")).segment(state)
    assert [s.label for s in seg.spans] == ["$.note"]


def test_runt_fragments_are_merged_into_neighbours() -> None:
    seg = SentenceChunker(min_chars=40).segment(TICKET)
    assert all(len(s.text) >= 20 for s in seg.spans)
    assert seg.render(_all(seg)) == TICKET


def test_span_cap_merges_rather_than_truncates() -> None:
    """Dropping spans past the cap would silently leave content unattributed."""
    text = " ".join(f"Sentence number {i} says something." for i in range(40))
    seg = SentenceChunker(min_chars=1, max_spans=8).segment(text)
    assert len(seg.spans) <= 8
    assert seg.render(_all(seg)) == text


def test_auto_chunker_matches_the_shape_of_the_state() -> None:
    assert isinstance(auto_chunker({"a": 1}), JsonLeafChunker)
    assert isinstance(auto_chunker("one sentence."), SentenceChunker)
    assert isinstance(auto_chunker("para one.\n\n" + "x" * 2100), BlockChunker)


def test_resolve_chunker_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="unknown chunker"):
        resolve_chunker("telepathy", "text")


def test_text_chunkers_reject_structured_state() -> None:
    with pytest.raises(TypeError, match="needs a string state"):
        SentenceChunker().segment({"a": 1})
