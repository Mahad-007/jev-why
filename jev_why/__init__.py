"""jev-why: causal attribution and calibration for TypeSafe Jev decisions.

Jev tells you what it decided. jev-why tells you why, and whether you should
believe it.

    from jev_why import explain
    from jev_why.questions import Noul

    exp = explain(page, {"is_injection": Noul(instructions="...")}, budget=0.05)
    print(exp["is_injection"].top(5))
"""

from __future__ import annotations

from jev_why.attribution import explain, explain_async
from jev_why.budget import Budget, BudgetExceeded, amortisation_factor, estimate_cost
from jev_why.cache import MemoryCache, NullCache, SqliteCache
from jev_why.chunking import (
    BlockChunker,
    JsonLeafChunker,
    SentenceChunker,
)
from jev_why.client import MissingApiKey, ModelVersionDrift, OfflineError
from jev_why.faithfulness import FaithfulnessReport
from jev_why.questions import Choice, Noul, QuestionError, Score
from jev_why.types import (
    Attribution,
    Explanation,
    MaskMode,
    QuestionExplanation,
    Span,
    Spend,
)

__version__ = "0.1.0"

__all__ = [
    "Attribution",
    "BlockChunker",
    "Budget",
    "BudgetExceeded",
    "Choice",
    "Explanation",
    "FaithfulnessReport",
    "JsonLeafChunker",
    "MaskMode",
    "MemoryCache",
    "MissingApiKey",
    "ModelVersionDrift",
    "Noul",
    "NullCache",
    "OfflineError",
    "QuestionError",
    "QuestionExplanation",
    "Score",
    "SentenceChunker",
    "Span",
    "Spend",
    "SqliteCache",
    "__version__",
    "amortisation_factor",
    "estimate_cost",
    "explain",
    "explain_async",
]
