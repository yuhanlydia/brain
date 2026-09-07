"""Canonical pronunciation identities shared across neural-evidence modules."""

from __future__ import annotations

from collections.abc import Sequence

Pronunciation = str | Sequence[str]


def canonicalize_pronunciation(value: Pronunciation) -> tuple[str, ...]:
    """Return a validated token tuple without splitting sequence elements."""

    if isinstance(value, str):
        raw_tokens: Sequence[object] = value.split()
    elif isinstance(value, Sequence):
        raw_tokens = value
    else:
        raise TypeError("pronunciation must be a string or sequence of strings")

    if not raw_tokens:
        raise ValueError("pronunciation must contain at least one token")

    tokens: list[str] = []
    for token in raw_tokens:
        if not isinstance(token, str):
            raise TypeError("pronunciation tokens must be strings")
        normalized = token.strip()
        if not normalized:
            raise ValueError("pronunciation tokens must be non-empty strings")
        tokens.append(normalized)
    return tuple(tokens)


__all__ = ["Pronunciation", "canonicalize_pronunciation"]
