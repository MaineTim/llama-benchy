import subprocess
import sys

import pytest

from llama_benchy.promptfile import PromptStore, sha256_text
from llama_benchy.prompts import PromptGenerator

TEXT = "0123456789" * 1000  # 10000 chars of digits


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


def _make_store(tmp_path, **overrides):
    kwargs = dict(
        book_url="https://example.com/book.txt",
        source_sha256=sha256_text(TEXT),
        text=TEXT,
        tokenizer=_FakeTokenizer(),
        tokenizer_name="fake/tokenizer",
        chars_per_token=1,
    )
    kwargs.update(overrides)
    return PromptStore(str(tmp_path / "family"), **kwargs)


def _make_generator(prompt_store=None):
    return PromptGenerator(_FakeCorpus(), prompt_store=prompt_store)


def test_generate_batch_non_fixed_varies_between_calls():
    import numpy as np

    np.random.seed(0)
    gen = _make_generator()
    batch1 = gen.generate_batch(1, 16, 0, no_cache=False)
    batch2 = gen.generate_batch(1, 16, 0, no_cache=False)
    # Random start index means batches differ between calls
    assert batch1 != batch2


def test_generate_batch_fixed_uses_pinned_slice(tmp_path):
    gen = _make_generator(prompt_store=_make_store(tmp_path))
    batch1 = gen.generate_batch(3, 16, 4, no_cache=False, fixed=True)
    batch2 = gen.generate_batch(3, 16, 4, no_cache=False, fixed=True)
    assert batch1 == batch2
    # chars_per_token=1: 4 context chars + 16 prompt chars
    assert all(c == TEXT[:4] and p == TEXT[4:20] for c, p in batch1)


def test_generate_batch_fixed_slices_are_shape_specific(tmp_path):
    # Regression: every (pp, depth) shape must send its own pinned text.
    # With one shared slice, pp16384 used to send the pp2048 text and the
    # reported throughput was inflated by the size ratio.
    gen = _make_generator(prompt_store=_make_store(tmp_path))
    small = gen.generate_batch(1, 16, 4, no_cache=False, fixed=True)
    large = gen.generate_batch(1, 32, 8, no_cache=False, fixed=True)
    assert [len(p) for _, p in small] == [16]
    assert [len(p) for _, p in large] == [32]
    assert [len(c) for c, _ in small] == [4]
    assert [len(c) for c, _ in large] == [8]


def test_generate_batch_fixed_empty_context_when_depth_zero(tmp_path):
    gen = _make_generator(prompt_store=_make_store(tmp_path))
    batch = gen.generate_batch(1, 16, 0, no_cache=False, fixed=True)
    assert batch == [("", TEXT[:16])]


def test_generate_batch_fixed_requires_prompt_store():
    gen = _make_generator()
    with pytest.raises(RuntimeError, match="no prompt store"):
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
            "/tmp/nonexistent-prompt-dir",
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
