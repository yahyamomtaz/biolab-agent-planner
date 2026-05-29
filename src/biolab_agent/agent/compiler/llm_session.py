"""LLMSession

Both the Planner and the LLM-backed Renderers do the same dance: call the
chat model with a system prompt, parse the JSON reply, validate it against a
Pydantic schema, and on failure reprompt with the validation error.
"""

from __future__ import annotations

import json
import re
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from biolab_agent.llm.base import ChatClient
from biolab_agent.logging import get_logger

log = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

_JSON_PATTERNS = (
    re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL),
    re.compile(r"(\{.*\})", re.DOTALL),
)


def _extract_json(raw: str) -> dict[str, Any] | None:
    if not raw:
        return None
    for pat in _JSON_PATTERNS:
        m = pat.search(raw)
        if not m:
            continue
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
    return None


class LLMSession:
    """A short-lived chat conversation that yields a validated Pydantic model.

    Designed to be constructed once per request and discarded. The message
    history mutation is intentional: callers append observations to the same
    session as the conversation evolves.
    """

    def __init__(
        self,
        client: ChatClient,
        model: str,
        *,
        system_prompt: str,
        temperature: float,
        num_predict: int,
        top_p: float,
        max_attempts: int = 2,
    ) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature
        self._num_predict = num_predict
        self._top_p = top_p
        self._max_attempts = max_attempts
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
        ]

    def add_user(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def complete(
        self,
        schema: type[T],
        *,
        constrained_decoding: bool = True,
    ) -> T | None:
        """Run the LLM until it yields a value valid under ``schema``.

        Returns ``None`` if all attempts fail; callers decide on a fallback.
        Each failed attempt is appended to the conversation along with a
        targeted retry prompt so the model can self-correct.
        """
        options: dict[str, Any] = {
            "temperature": self._temperature,
            "num_predict": self._num_predict,
            "top_p": self._top_p,
        }
        if constrained_decoding:
            options["format"] = schema.model_json_schema()

        schema_name = schema.__name__
        last_raw: str | None = None
        last_problem: str | None = None
        for attempt in range(self._max_attempts):
            try:
                resp = self._client.chat(
                    model=self._model,
                    messages=self.messages,
                    options=options,
                )
            except (RuntimeError, ConnectionError, TimeoutError) as exc:
                log.warning(
                    "llm_session.chat_failed",
                    schema=schema_name, attempt=attempt, error=str(exc),
                )
                return None
            try:
                raw = resp["message"]["content"]
            except (KeyError, TypeError):
                log.warning(
                    "llm_session.malformed_response",
                    schema=schema_name, attempt=attempt,
                )
                return None

            last_raw = raw
            parsed = _extract_json(raw)
            if parsed is None:
                last_problem = "no JSON object found in reply"
                self.messages.append({"role": "assistant", "content": raw})
                self.messages.append(
                    {"role": "user", "content": "Return exactly one JSON object."},
                )
                continue
            try:
                return schema.model_validate(parsed)
            except ValidationError as exc:
                last_problem = str(exc.errors())
                self.messages.append({"role": "assistant", "content": raw})
                self.messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Validation failed: {exc.errors()}. "
                            "Re-emit one JSON object matching the schema."
                        ),
                    },
                )
        # All attempts exhausted — log enough to diagnose without dumping
        # the full transcript every time.
        log.warning(
            "llm_session.exhausted_attempts",
            schema=schema_name,
            attempts=self._max_attempts,
            last_problem=last_problem,
            last_raw=(last_raw or "")[:800],
        )
        return None
