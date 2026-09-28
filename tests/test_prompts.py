import subprocess
import sys

import pytest

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


def _make_generator():
    return PromptGenerator(_FakeCorpus())


def test_generate_batch_non_fixed_varies_between_calls():
    import numpy as np

    np.random.seed(0)
    gen = _make_generator()
    batch1 = gen.generate_batch(1, 16, 0, no_cache=False)
    batch2 = gen.generate_batch(1, 16, 0, no_cache=False)
    # Random start index means batches differ between calls
    assert batch1 != batch2


def test_generate_batch_fixed_reuses_same_prompts():
    gen = _make_generator()
    batch1 = gen.generate_batch(2, 16, 4, no_cache=False, fixed=True)
    batch2 = gen.generate_batch(2, 16, 4, no_cache=False, fixed=True)
    assert batch1 == batch2


def test_generate_batch_fixed_caches_per_shape():
    gen = _make_generator()
    small = gen.generate_batch(1, 16, 0, no_cache=False, fixed=True)
    large = gen.generate_batch(1, 32, 0, no_cache=False, fixed=True)
    assert [p for _, p in small] != [p for _, p in large]
    # Original fixed batch is still cached
    again = gen.generate_batch(1, 16, 0, no_cache=False, fixed=True)
    assert again == small


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
