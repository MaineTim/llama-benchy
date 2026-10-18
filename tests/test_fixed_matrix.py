"""End-to-end regression for --fixed-prompt across a pp x depth matrix.

The original bug: a single pinned slice was shared by every test shape, so
pp16384/d32768 sent the text pinned for pp2048 while the metrics were
computed against the nominal sizes — inflating t/s by the size ratio.
These tests drive the actual BenchmarkRunner and assert that every shape
sends its own pinned payload sizes.
"""

import pytest

from llama_benchy.client import RequestResult
from llama_benchy.config import BenchmarkConfig
from llama_benchy.promptfile import PromptStore, sha256_text
from llama_benchy.prompts import PromptGenerator
from llama_benchy.runner import BenchmarkRunner

TEXT = "0123456789" * 1000  # 10000 chars


class _CharTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(range(len(text)))


class _FakeCorpus:
    def get_tokenizer(self):
        return _CharTokenizer()

    def get_tokens(self):
        return list(range(1000))


class _RecordingClient:
    def __init__(self):
        self.payloads = []

    async def warmup(self, session, tokenizer=None):
        return 0, 0

    async def run_coherence_test(self, session):
        return True

    async def measure_latency(self, session, mode="api", warmup_runs=1, measured_runs=3):
        return 0.001

    async def run_generation(
        self,
        session,
        context_text,
        prompt_text,
        max_tokens,
        no_cache,
        tokenizer=None,
        progress=None,
        request_id=None,
    ):
        self.payloads.append((len(context_text), len(prompt_text)))
        return RequestResult(
            start_ts=1.0,
            first_token_ts=1.1,
            first_response_ts=1.05,
            end_ts=1.3,
            prompt_tokens=4,
            total_tokens=2,
            token_timestamps=[1.1, 1.2],
        )


def _make_config():
    return BenchmarkConfig(
        base_url="http://example.test/v1",
        api_key="EMPTY",
        model="model",
        served_model_name="model",
        tokenizer=None,
        pp_counts=[8, 16],
        tg_counts=[4],
        exact_tg=False,
        depths=[0, 32],
        num_runs=1,
        warmup_runs=1,
        short_warmup=True,
        no_cache=False,
        fixed_prompt=True,
        latency_mode="none",
        no_warmup=False,
        skip_coherence=True,
        adapt_prompt=True,
        enable_prefix_caching=False,
        book_url="",
        post_run_cmd=None,
        concurrency_levels=[1],
        save_result=None,
        result_format="json",
        save_total_throughput_timeseries=False,
        save_all_throughput_timeseries=False,
        exit_on_first_fail=False,
        no_results_on_fail=False,
        extra_body={},
        emit_progress=None,
    )


@pytest.mark.asyncio
async def test_fixed_prompt_matrix_sends_distinct_payload_per_shape(tmp_path):
    store = PromptStore(
        str(tmp_path / "family"),
        book_url="https://example.com/book.txt",
        source_sha256=sha256_text(TEXT),
        text=TEXT,
        tokenizer=_CharTokenizer(),
        tokenizer_name="fake/tokenizer",
        chars_per_token=1,
    )
    generator = PromptGenerator(_FakeCorpus(), prompt_store=store)
    client = _RecordingClient()

    runner = BenchmarkRunner(_make_config(), client, generator)
    await runner.run_suite()

    # chars_per_token=1 => (context chars, prompt chars) per shape:
    # d0/pp8 -> (0, 8); d0/pp16 -> (0, 16);
    # d32/pp8 -> (32, 8); d32/pp16 -> (32, 16).
    # Each shape runs twice (one warmup, one measured).
    assert set(client.payloads) == {
        (0, 8),
        (0, 16),
        (32, 8),
        (32, 16),
    }
    assert len(client.payloads) == 8

    # And the pinned text is stable across the warmup and measured run of
    # the same shape: re-deriving it from the store yields the same sizes.
    for (pp, depth), slice_ in [((8, 0), store.get(8, 0)), ((16, 32), store.get(16, 32))]:
        assert len(slice_.prompt.prompt) == pp
        assert len(slice_.prompt.context) == depth


def test_char_tokenizer_cut_is_deterministic(tmp_path):
    # Same text + same shape in a fresh store (same dir) loads identical text.
    def fresh():
        return PromptStore(
            str(tmp_path / "family"),
            book_url="u",
            source_sha256=sha256_text(TEXT),
            text=TEXT,
            tokenizer=_CharTokenizer(),
            chars_per_token=1,
        )

    a = fresh().get(16, 4).prompt.prompt
    b = fresh().get(16, 4).prompt.prompt
    assert a == b
    assert sha256_text(a) == sha256_text(b)
