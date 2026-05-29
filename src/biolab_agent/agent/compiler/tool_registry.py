"""ToolRegistry — dependency-injectable tool table.

Replaces the previous pattern of mutating ``biolab_agent.tools.TOOL_IMPLS``
in agent ``__init__``. Each registry instance carries its own mapping, so
tests can build a registry with stubbed tools without affecting global state.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class ToolRegistry:
    """A name -> callable mapping plus the matching Ollama function specs."""

    def __init__(
        self,
        impls: dict[str, Callable[..., Any]],
        specs: list[dict[str, Any]],
    ) -> None:
        self._impls = dict(impls)
        self._specs = list(specs)

    @classmethod
    def default(cls) -> ToolRegistry:
        """The production registry: real tools + the optimized catalog lookup.

        Importing ``biolab_agent.tools`` lazily so test runs that stub tools
        out don't pay for the production deps (qdrant client, etc.).
        """
        from biolab_agent.tools import TOOL_IMPLS, TOOL_SPECS
        from biolab_agent.tools.lookup_reagents import (
            lookup_reagent as optimized_lookup_reagent,
        )

        impls = {**TOOL_IMPLS, "lookup_reagent": optimized_lookup_reagent}
        return cls(impls=impls, specs=TOOL_SPECS)

    def get(self, name: str) -> Callable[..., Any]:
        try:
            return self._impls[name]
        except KeyError as exc:
            raise KeyError(
                f"Unknown tool {name!r}. Known tools: {sorted(self._impls)}",
            ) from exc

    @property
    def names(self) -> list[str]:
        return list(self._impls)

    @property
    def specs(self) -> list[dict[str, Any]]:
        return list(self._specs)

    def with_override(
        self,
        name: str,
        impl: Callable[..., Any],
    ) -> ToolRegistry:
        """Return a new registry with one tool replaced. Tests use this."""
        return ToolRegistry(impls={**self._impls, name: impl}, specs=self._specs)
