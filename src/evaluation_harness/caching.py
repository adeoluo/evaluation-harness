"""A caching wrapper for models.

Re-running an evaluation - after changing a grader, adding a metric, or debugging - should
not pay to regenerate predictions that have not changed. :class:`CachingModel` wraps any
model and memoizes predictions keyed by the example identifier, optionally persisting the
cache to a JSON file so it survives across processes.

The wrapper is itself a :class:`~evaluation_harness.core.model.Model`, so it drops into the
runner transparently.
"""

from __future__ import annotations

import json
from pathlib import Path

from evaluation_harness.core.example import Example
from evaluation_harness.core.model import Model


class CachingModel:
    """Memoizes a wrapped model's predictions by example identifier."""

    def __init__(self, model: Model, *, cache_path: Path | None = None) -> None:
        """Wrap ``model`` with a prediction cache.

        Args:
            model: The underlying model whose predictions are cached.
            cache_path: Optional JSON file to load from and persist to. When provided and
                existing, the cache is seeded from it; call :meth:`save` to write back.
        """
        self._model = model
        self._cache_path = cache_path
        self._cache: dict[str, str] = {}
        self._hits = 0
        self._misses = 0
        if cache_path is not None and cache_path.exists():
            self._cache = json.loads(cache_path.read_text(encoding="utf-8"))

    @property
    def name(self) -> str:
        return self._model.name

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    def generate(self, example: Example) -> str:
        key = example.identifier
        if key in self._cache:
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        prediction = self._model.generate(example)
        self._cache[key] = prediction
        return prediction

    def save(self) -> None:
        """Persist the cache to ``cache_path`` if one was provided."""
        if self._cache_path is None:
            raise ValueError("CachingModel has no cache_path to save to")
        self._cache_path.write_text(json.dumps(self._cache, indent=2), encoding="utf-8")
