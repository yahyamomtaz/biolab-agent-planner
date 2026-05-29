"""Tools the agent registers with the LLM.

Each module exposes:
    - a plain Python callable with a typed signature
    - a ``TOOL_SPEC`` dict in Ollama function-calling schema

The baseline agent imports ``TOOL_SPECS`` and ``TOOL_IMPLS`` from here; an
alternative agent can replace either map with its own callables.
"""

from biolab_agent.tools.compose import compose_protocol, compose_protocol_spec
from biolab_agent.tools.rag_hybrid import retrieve_protocol, retrieve_protocol_spec
from biolab_agent.tools.reagents import lookup_reagent, lookup_reagent_spec
from biolab_agent.tools.segment import segment_wells, segment_wells_spec

TOOL_IMPLS = {
    "segment_wells": segment_wells,
    "retrieve_protocol": retrieve_protocol,
    "lookup_reagent": lookup_reagent,
    "compose_protocol": compose_protocol,
}

TOOL_SPECS = [
    segment_wells_spec,
    retrieve_protocol_spec,
    lookup_reagent_spec,
    compose_protocol_spec,
]

__all__ = [
    "TOOL_IMPLS",
    "TOOL_SPECS",
    "compose_protocol",
    "lookup_reagent",
    "retrieve_protocol",
    "segment_wells",
]
