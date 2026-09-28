import pytest

from llama_benchy.client import RequestResult
from llama_benchy.config import BenchmarkConfig
from llama_benchy.runner import BenchmarkRunner


class _FakeCorpus:
    def get_tokenizer(self):
        return None


class _RecordingPromptGenerator:
    corpus = _FakeCorpus()

    def __init__(self):
        self.pp_calls = []

    def generate_batch(self, concurrency, pp, depth, no_cache, fixed=False):
        self.pp_calls.append(pp)
        return [("", "hello") for _ in range(concurrency)]


class _RecordingClient:
    def __init__(self):
        self.max_tokens_calls = []

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
        self.max_tokens_calls.append(max_tokens)
        return RequestResult(
            start_ts=1.0,
            first_token_ts=1.1,
            end_ts=1.3,
            prompt_tokens=4,
            total_tokens=2,
            token_timestamps=[1.1, 1.2],
        )


def _make_config(pp, tg, short_warmup):
    return BenchmarkConfig(
        base_url="http://example.test/v1",
        api_key="EMPTY",
        model="model",
        served_model_name="model",
        tokenizer=None,
        pp_counts=[pp],
        tg_counts=[tg],
        exact_tg=False,
        depths=[0],
        num_runs=1,
        warmup_runs=1,
        short_warmup=short_warmup,
        no_cache=False,
        latency_mode="none",
        no_warmup=False,
        skip_coherence=True,
        adapt_prompt=False,
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
async def test_short_warmup_caps_large_shapes(tmp_path):
    config = _make_config(pp=4096, tg=1024, short_warmup=True)
    generator = _RecordingPromptGenerator()
    client = _RecordingClient()

    runner = BenchmarkRunner(config, client, generator)
    await runner.run_suite()

    # warmup run first (capped), measured run second (full size)
    assert generator.pp_calls == [2048, 4096]
    assert client.max_tokens_calls == [512, 1024]


@pytest.mark.asyncio
async def test_short_warmup_only_tg_exceeds_limit(tmp_path):
    config = _make_config(pp=1024, tg=1024, short_warmup=True)
    generator = _RecordingPromptGenerator()
    client = _RecordingClient()

    runner = BenchmarkRunner(config, client, generator)
    await runner.run_suite()

    # pp is under the cap, so warmup pp is unchanged; tg is capped
    assert generator.pp_calls == [1024, 1024]
    assert client.max_tokens_calls == [512, 1024]


@pytest.mark.asyncio
async def test_short_warmup_untouched_for_small_shapes(tmp_path):
    config = _make_config(pp=1024, tg=256, short_warmup=True)
    generator = _RecordingPromptGenerator()
    client = _RecordingClient()

    runner = BenchmarkRunner(config, client, generator)
    await runner.run_suite()

    assert generator.pp_calls == [1024, 1024]
    assert client.max_tokens_calls == [256, 256]


@pytest.mark.asyncio
async def test_no_short_warmup_keeps_full_size(tmp_path):
    config = _make_config(pp=4096, tg=1024, short_warmup=False)
    generator = _RecordingPromptGenerator()
    client = _RecordingClient()

    runner = BenchmarkRunner(config, client, generator)
    await runner.run_suite()

    assert generator.pp_calls == [4096, 4096]
    assert client.max_tokens_calls == [1024, 1024]
