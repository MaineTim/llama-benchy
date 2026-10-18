"""Persistence for fixed prompts (cross-session reproducibility).

A fixed prompt is a slice of the source book text, cut by explicit
character offsets (not tokens, not a random seed) and pinned to a JSON
file. Later sessions read the same file, so the exact same text is sent
by construction, regardless of tokenizer or machine.

Because a benchmark suite runs many shapes (``--pp`` x ``--depth``), one
pinned slice is not enough: every requested ``(pp, depth)`` shape needs
its own slice, otherwise a ``pp16384`` test would silently send the text
that was pinned for ``pp2048`` and the throughput would be inflated by
the ratio. Slices therefore live in a *family directory*, one file per
shape, named ``ctx<depth>_pp<pp>.json``.

File format (``llama-benchy-prompt.v2``)::

    {
      "schema": "llama-benchy-prompt.v2",
      "book_url": "https://...",
      "source_sha256": "<sha256 of the full book text>",
      "offset": 10000,
      "pp": 16384,
      "depth": 0,
      "context_length": 0,
      "prompt_length": 65536,
      "prompt_tokens": 16390,
      "tokenizer": "Qwen/Qwen3-8B",
      "context": "",
      "prompt": "...",
      "created_at": "2025-01-01T00:00:00Z"
    }

The ``context`` slice is the text immediately before the ``prompt`` slice
(both cut from one contiguous region), so the prompt follows the context
exactly — which matters for ``--enable-prefix-caching`` runs.
``pp``/``depth`` record the nominal test shape the slice was cut for;
``prompt_tokens`` is the measured token count at creation time (metadata
only — the pinned *text* remains the source of truth, since the cut is
character-based and token counts legitimately differ across tokenizers).

Legacy ``llama-benchy-prompt.v1`` files (a single slice, no shape fields)
are still readable. Such a file can only serve the shape it matches
(prompt token count within 15% of the requested ``pp`` and matching
presence of a context slice); anything else gets its own v2 slice.
"""

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from typing import NamedTuple, Optional

from pydantic import BaseModel, ConfigDict, Field, model_serializer

SCHEMA_VERSION = "llama-benchy-prompt.v2"
SUPPORTED_SCHEMA_PREFIX = "llama-benchy-prompt.v"

DEFAULT_CHARS_PER_TOKEN = 4
# Deviation between the nominal token size of a shape and the measured
# token count of the pinned character slice that triggers a warning.
TOKEN_COUNT_TOLERANCE = 0.15


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class PromptMismatchError(ValueError):
    """Explicitly requested slice parameters conflict with a pinned file."""


class FixedPrompt(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    # JSON key is "schema" (alias); the attribute is schema_version to avoid
    # shadowing pydantic's deprecated BaseModel.schema attribute.
    schema_version: str = Field(default=SCHEMA_VERSION, alias="schema", description="Manifest schema version")
    book_url: str = Field(..., description="URL of the source text the slice was cut from")
    source_sha256: str = Field(..., description="SHA-256 of the full source text at creation time")
    offset: int = Field(..., description="Character offset of the slice in the source text")
    context_length: int = Field(..., description="Length of the context slice in characters")
    prompt_length: int = Field(..., description="Length of the prompt slice in characters")
    context: str = Field(..., description="Pinned context text (sent before the prompt)")
    prompt: str = Field(..., description="Pinned prompt text")
    created_at: str = Field(..., description="UTC timestamp when the file was written")
    # v2 additions (optional so v1 files keep loading)
    pp: Optional[int] = Field(None, description="Nominal prompt token count of the test shape this slice serves")
    depth: Optional[int] = Field(None, description="Nominal context depth of the test shape this slice serves")
    prompt_tokens: Optional[int] = Field(None, description="Measured prompt token count at creation time (metadata only)")
    tokenizer: Optional[str] = Field(None, description="Tokenizer id used to measure prompt_tokens")

    @model_serializer(mode="wrap")
    def _serialize(self, handler):
        data = handler(self)
        # Keep v1 files byte-compatible: omit v2 fields that are unset.
        for key in ("pp", "depth", "prompt_tokens", "tokenizer"):
            if data.get(key) is None:
                data.pop(key, None)
        return data


class PromptSlice(NamedTuple):
    prompt: FixedPrompt
    status: str  # "created" | "loaded" | "regenerated" | "legacy"
    previous_sha256: Optional[str]  # prompt sha256 replaced by --new-fixed-prompt
    warning: Optional[str]  # non-fatal issue to surface to the user (never fatal twice)


def slice_text(text: str, offset: int, context_length: int, prompt_length: int):
    """Cut (context, prompt) from ``text`` by character offsets.

    The context and prompt are one contiguous region:
    ``text[offset : offset + context_length + prompt_length]``.
    """
    if offset < 0:
        raise ValueError(f"offset must be >= 0, got {offset}")
    if context_length < 0:
        raise ValueError(f"context_length must be >= 0, got {context_length}")
    if prompt_length <= 0:
        raise ValueError(f"prompt_length must be > 0, got {prompt_length}")
    end = offset + context_length + prompt_length
    if end > len(text):
        raise ValueError(
            f"slice exceeds source text: offset={offset} + context_length={context_length} "
            + f"+ prompt_length={prompt_length} = {end} chars, but text has only {len(text)} chars"
        )
    p_start = offset + context_length
    return text[offset:p_start], text[p_start:end]


def measure_tokens(tokenizer, text: str) -> Optional[int]:
    if tokenizer is None or not text:
        return None
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except Exception:
        return None


def slice_filename(pp: int, depth: int) -> str:
    return f"ctx{depth}_pp{pp}.json"


def _write_file(path: str, prompt: FixedPrompt) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(prompt.model_dump(by_alias=True), f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp_path, path)


def _read_file(path: str) -> FixedPrompt:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    version = data.get("schema", "")
    if not version.startswith(SUPPORTED_SCHEMA_PREFIX):
        raise ValueError(f"{path} is not a llama-benchy prompt file (unknown schema '{version}')")
    return FixedPrompt(**data)


class PromptStore:
    """Per-shape pinned slices for one fixed-prompt family.

    The store hands out the pinned (context, prompt) slice for a given
    nominal ``(pp, depth)`` test shape. Slices are cut by character
    budgets derived from the requested token sizes
    (``chars_per_token * tokens``) and persisted to one JSON file per
    shape under ``family_dir``, so later sessions reuse the identical
    text. The file is the source of truth: once a shape's slice exists,
    creation-time parameters no longer matter (except ``--new``, which
    regenerates after backing the old file up to ``<file>.bak``).
    """

    def __init__(
        self,
        family_dir: str,
        *,
        book_url: str,
        source_sha256: str,
        text: str,
        tokenizer=None,
        tokenizer_name: Optional[str] = None,
        offset: int = 0,
        chars_per_token: int = DEFAULT_CHARS_PER_TOKEN,
        prompt_length: Optional[int] = None,
        context_length: Optional[int] = None,
        new: bool = False,
        legacy_path: Optional[str] = None,
        explicit_offset: bool = False,
        explicit_prompt_length: bool = False,
        explicit_context_length: bool = False,
    ) -> None:
        self.family_dir = family_dir
        self.book_url = book_url
        self.source_sha256 = source_sha256
        self.text = text
        self.tokenizer = tokenizer
        self.tokenizer_name = tokenizer_name
        self.offset = offset
        self.chars_per_token = chars_per_token
        self.prompt_length = prompt_length
        self.context_length = context_length
        self.new = new
        self.legacy_path = legacy_path
        self.explicit_offset = explicit_offset
        self.explicit_prompt_length = explicit_prompt_length
        self.explicit_context_length = explicit_context_length
        self._cache = {}
        self._legacy = None
        self._legacy_checked = False

    # -- internals ---------------------------------------------------------

    def _budgets(self, pp: int, depth: int):
        prompt_chars = self.prompt_length if self.prompt_length is not None else self.chars_per_token * pp
        if depth <= 0:
            context_chars = 0
        elif self.context_length is not None:
            context_chars = self.context_length
        else:
            context_chars = self.chars_per_token * depth
        return context_chars, prompt_chars

    def _cut(self, pp: int, depth: int) -> FixedPrompt:
        context_chars, prompt_chars = self._budgets(pp, depth)
        context, prompt_text = slice_text(self.text, self.offset, context_chars, prompt_chars)
        return FixedPrompt(
            book_url=self.book_url,
            source_sha256=self.source_sha256,
            offset=self.offset,
            pp=pp,
            depth=max(0, depth),
            context_length=context_chars,
            prompt_length=prompt_chars,
            prompt_tokens=measure_tokens(self.tokenizer, prompt_text),
            tokenizer=self.tokenizer_name,
            context=context,
            prompt=prompt_text,
            created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )

    def _check_explicit_params(self, path: str, prompt: FixedPrompt) -> None:
        """Abort if explicitly requested slice params conflict with the file.

        The file is the source of truth; silently regenerating on a param
        change would reintroduce cross-session drift, so mismatches are a
        hard error (use a different --prompt-path or --new-fixed-prompt).
        """
        if self.explicit_offset and self.offset != prompt.offset:
            raise PromptMismatchError(
                f"--prompt-offset {self.offset} does not match pinned slice {path} (offset {prompt.offset})"
            )
        if self.explicit_prompt_length and self.prompt_length != prompt.prompt_length:
            raise PromptMismatchError(
                f"--prompt-length {self.prompt_length} does not match pinned slice {path} "
                f"(length {prompt.prompt_length})"
            )
        if self.explicit_context_length and self.context_length not in (None, prompt.context_length):
            if prompt.context_length > 0:
                raise PromptMismatchError(
                    f"--prompt-context-length {self.context_length} does not match pinned slice {path} "
                    f"(length {prompt.context_length})"
                )

    def _token_count_warning(self, label: str, prompt: FixedPrompt) -> Optional[str]:
        nominal = prompt.pp
        measured = prompt.prompt_tokens
        if measured is None:
            measured = measure_tokens(self.tokenizer, prompt.prompt)
        if nominal is None or not nominal or measured is None:
            return None
        deviation = abs(measured - nominal) / nominal
        if deviation > TOKEN_COUNT_TOLERANCE:
            return (
                f"{label}: pinned slice measures {measured} tokens but the test nominal size is "
                f"{nominal} tokens ({deviation * 100:.0f}% off; {prompt.prompt_length} chars). "
                "Metrics are computed against the nominal size; consider a different "
                "--prompt-chars-per-token when creating a new family."
            )
        return None

    def _source_warning(self, path: str, prompt: FixedPrompt) -> Optional[str]:
        if prompt.source_sha256 != self.source_sha256:
            return (
                f"{path} was cut from a different source text (file sha256: "
                f"{prompt.source_sha256}, current text sha256: {self.source_sha256}). "
                "The file content is used as-is."
            )
        return None

    def _load_legacy(self) -> Optional[FixedPrompt]:
        if not self._legacy_checked:
            self._legacy_checked = True
            if self.legacy_path and os.path.exists(self.legacy_path):
                try:
                    self._legacy = _read_file(self.legacy_path)
                except Exception as e:
                    print(f"Warning: ignoring unreadable legacy prompt file {self.legacy_path}: {e}")
                    self._legacy = None
        return self._legacy

    def _legacy_slice(self, pp: int, depth: int) -> Optional[PromptSlice]:
        """Use a pre-v2 single-slice file if it matches this shape."""
        legacy = self._load_legacy()
        if legacy is None:
            return None
        wants_context = depth > 0
        has_context = bool(legacy.context)
        if wants_context != has_context:
            return None
        measured = legacy.prompt_tokens
        if measured is None:
            measured = measure_tokens(self.tokenizer, legacy.prompt)
        if measured is None or pp <= 0:
            return None
        if abs(measured - pp) / pp > TOKEN_COUNT_TOLERANCE:
            return None
        warning = (
            f"using legacy single-slice prompt file {self.legacy_path} for shape pp{pp}@d{depth} "
            "(only shapes matching the pinned slice are served from it)"
        )
        source_warning = self._source_warning(self.legacy_path, legacy)
        if source_warning:
            warning = f"{warning}; {source_warning}"
        return PromptSlice(prompt=legacy, status="legacy", previous_sha256=None, warning=warning)

    # -- public API ----------------------------------------------------------

    def get(self, pp: int, depth: int) -> PromptSlice:
        """Return the pinned slice for the nominal (pp, depth) shape.

        - File missing -> cut the slice for this shape, write it, status "created".
        - File present -> read it as-is, status "loaded".
        - File present + new=True -> back up to ``<path>.bak``, re-cut, status "regenerated".
        - Pre-v2 file (legacy_path) matching this shape -> status "legacy" with a warning.
        """
        key = (pp, max(0, depth))
        if key in self._cache:
            return self._cache[key]

        if self.legacy_path and not self.new:
            legacy_slice = self._legacy_slice(pp, depth)
            if legacy_slice is not None:
                self._cache[key] = legacy_slice
                return legacy_slice

        path = os.path.join(self.family_dir, slice_filename(pp, depth))
        if not os.path.exists(path):
            prompt = self._cut(pp, depth)
            _write_file(path, prompt)
            result = PromptSlice(prompt=prompt, status="created", previous_sha256=None, warning=None)
        else:
            existing = _read_file(path)
            self._check_explicit_params(path, existing)
            if not self.new:
                result = PromptSlice(prompt=existing, status="loaded", previous_sha256=None, warning=None)
            else:
                previous_sha256 = sha256_text(existing.prompt)
                shutil.copy2(path, path + ".bak")
                prompt = self._cut(pp, depth)
                _write_file(path, prompt)
                result = PromptSlice(
                    prompt=prompt, status="regenerated", previous_sha256=previous_sha256, warning=None
                )

        warnings = [w for w in (
            self._source_warning(path, result.prompt),
            self._token_count_warning(f"pp{pp}@d{depth}", result.prompt),
        ) if w]
        if warnings:
            result = result._replace(warning="\n".join(warnings))

        self._cache[key] = result
        return result

    def shapes(self):
        return sorted(self._cache.keys())
