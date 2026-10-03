import subprocess
import sys

import pytest

from llama_benchy.promptfile import FixedPrompt, SCHEMA_VERSION
from llama_benchy.prompts import PromptGenerator


class _FakeTokenizer:
    """Extremely simple whitespace-ish tokenizer: one token per char."""

    def encode(self, text, add_special_tokens=False):
        return list(range(len(text)))

    def decode(self, token_ids):
        return "".join(str(i % 10) for i in token_ids)


class _FakeCorpus:
    def __init__(self):
        self._tokenizer = _FakeTokenizer()
        self._tokens = list(range(10, 1000))

    def get_tokenizer(self):
        return self._tokenizer

    def get_tokens(self):
        return self._tokens


def _make_generator(fixed_prompt=None):
    return PromptGenerator(_FakeCorpus(), fixed_prompt=fixed_prompt)


def _make_fixed_prompt(context="ctxtext", prompt="prompttext"):
    return FixedPrompt(
        schema_version=SCHEMA_VERSION,
        book_url="https://example.com/book.txt",
        source_sha256="0" * 64,
        offset=1000,
        context_length=len(context),
        prompt_length=len(prompt),
        context=context,
        prompt=prompt,
        created_at="2025-01-01T00:00:00Z",
    )


def test_generate_batch_non_fixed_varies_between_calls():
    import numpy as np

    np.random.seed(0)
    gen = _make_generator()
    batch1 = gen.generate_batch(1, 16, 0, no_cache=False)
    batch2 = gen.generate_batch(1, 16, 0, no_cache=False)
    # Random start index means batches differ between calls
    assert batch1 != batch2


def test_generate_batch_fixed_uses_prompt_file():
    gen = _make_generator(fixed_prompt=_make_fixed_prompt())
    batch1 = gen.generate_batch(3, 16, 4, no_cache=False, fixed=True)
    batch2 = gen.generate_batch(3, 16, 4, no_cache=False, fixed=True)
    assert batch1 == batch2
    assert all(c == "ctxtext" and p == "prompttext" for c, p in batch1)


def test_generate_batch_fixed_same_text_across_shapes():
    # The file is the anchor: every (pp, depth) shape sends the pinned text.
    gen = _make_generator(fixed_prompt=_make_fixed_prompt())
    small = gen.generate_batch(1, 16, 4, no_cache=False, fixed=True)
    large = gen.generate_batch(1, 32, 8, no_cache=False, fixed=True)
    assert [p for _, p in small] == [p for _, p in large]


def test_generate_batch_fixed_empty_context_when_depth_zero():
    gen = _make_generator(fixed_prompt=_make_fixed_prompt())
    batch = gen.generate_batch(1, 16, 0, no_cache=False, fixed=True)
    assert batch == [("", "prompttext")]


def test_generate_batch_fixed_requires_prompt_file():
    gen = _make_generator()
    with pytest.raises(RuntimeError, match="no prompt file"):
        gen.generate_batch(1, 16, 0, no_cache=False, fixed=True)


def test_fixed_prompt_conflicts_with_no_cache():
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "llama_benchy",
            "--base-url",
            "http://localhost:1",
            "--model",
            "foo/bar",
            "--fixed-prompt",
            "--no-cache",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "conflict" in proc.stderr


def test_prompt_path_implies_fixed_and_conflicts_with_no_cache():
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "llama_benchy",
            "--base-url",
            "http://localhost:1",
            "--model",
            "foo/bar",
            "--prompt-path",
            "/tmp/nonexistent-prompt.json",
            "--no-cache",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "conflict" in proc.stderr


def test_new_fixed_prompt_conflicts_with_no_cache():
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "llama_benchy",
            "--base-url",
            "http://localhost:1",
            "--model",
            "foo/bar",
            "--new-fixed-prompt",
            "--no-cache",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "conflict" in proc.stderr
