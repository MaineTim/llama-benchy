"""Persistence for fixed prompts (cross-session reproducibility).

A fixed prompt is a slice of the source book text, cut by explicit
character offsets (not tokens, not a random seed) and pinned to a JSON
file. Later sessions read the same file, so the exact same text is sent
by construction, regardless of tokenizer or machine.

File format (``llama-benchy-prompt.v1``)::

    {
      "schema": "llama-benchy-prompt.v1",
      "book_url": "https://...",
      "source_sha256": "<sha256 of the full book text>",
      "offset": 10000,
      "context_length": 4096,
      "prompt_length": 8192,
      "context": "...",
      "prompt": "...",
      "created_at": "2025-01-01T00:00:00Z"
    }

The ``context`` slice is the text immediately before the ``prompt`` slice
(both cut from one contiguous region), so the prompt follows the context
exactly — which matters for ``--enable-prefix-caching`` runs.
"""

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from typing import NamedTuple, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "llama-benchy-prompt.v1"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


class FixedPromptResult(NamedTuple):
    prompt: FixedPrompt
    status: str  # "created" | "loaded" | "regenerated"
    previous_sha256: Optional[str]  # prompt sha256 replaced by --new-fixed-prompt


def slice_text(text: str, offset: int, context_length: int, prompt_length: int) -> Tuple[str, str]:
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
            f"+ prompt_length={prompt_length} = {end} chars, but text has only {len(text)} chars"
        )
    p_start = offset + context_length
    return text[offset:p_start], text[p_start:end]


def check_explicit_params(
    prompt: FixedPrompt,
    offset: Optional[int] = None,
    prompt_length: Optional[int] = None,
    context_length: Optional[int] = None,
) -> Optional[str]:
    """Error message if explicitly requested slice params differ from the file.

    The file is the source of truth; silently regenerating on a param change
    would reintroduce cross-session drift, so mismatches are a hard error
    (use a different --prompt-path or --new-fixed-prompt instead).
    """
    if offset is not None and offset != prompt.offset:
        return f"--prompt-offset {offset} does not match prompt file (offset {prompt.offset})"
    if prompt_length is not None and prompt_length != prompt.prompt_length:
        return f"--prompt-length {prompt_length} does not match prompt file (length {prompt.prompt_length})"
    if context_length is not None and context_length != prompt.context_length:
        return (
            f"--prompt-context-length {context_length} does not match prompt file "
            f"(length {prompt.context_length})"
        )
    return None


def _write_file(path: str, prompt: FixedPrompt) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(prompt.model_dump(by_alias=True), f, ensure_ascii=False, indent=2)
        f.write("\n")


def _read_file(path: str) -> FixedPrompt:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    version = data.get("schema", "")
    if not version.startswith("llama-benchy-prompt.v"):
        raise ValueError(f"{path} is not a llama-benchy prompt file (unknown schema '{version}')")
    return FixedPrompt(**data)



def load_or_create(
    path: str,
    *,
    book_url: str,
    source_sha256: str,
    text: str,
    offset: int,
    context_length: int,
    prompt_length: int,
    new: bool = False,
) -> FixedPromptResult:
    """Return the fixed prompt for ``path``, creating the file when needed.

    - File missing            -> cut the slice from ``text``, write it, status "created".
    - File present             -> read it as-is, status "loaded" (slice params ignored).
    - File present + new=True  -> back the old file up to ``<path>.bak``, cut a fresh
      slice, overwrite, status "regenerated".
    """
    if not os.path.exists(path):
        context, prompt_text = slice_text(text, offset, context_length, prompt_length)
        prompt = FixedPrompt(
            book_url=book_url,
            source_sha256=source_sha256,
            offset=offset,
            context_length=context_length,
            prompt_length=prompt_length,
            context=context,
            prompt=prompt_text,
            created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        _write_file(path, prompt)
        return FixedPromptResult(prompt=prompt, status="created", previous_sha256=None)

    existing = _read_file(path)
    if not new:
        return FixedPromptResult(prompt=existing, status="loaded", previous_sha256=None)

    previous_sha256 = sha256_text(existing.prompt)
    backup_path = path + ".bak"
    shutil.copy2(path, backup_path)
    context, prompt_text = slice_text(text, offset, context_length, prompt_length)
    prompt = FixedPrompt(
        book_url=book_url,
        source_sha256=source_sha256,
        offset=offset,
        context_length=context_length,
        prompt_length=prompt_length,
        context=context,
        prompt=prompt_text,
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    _write_file(path, prompt)
    return FixedPromptResult(prompt=prompt, status="regenerated", previous_sha256=previous_sha256)
