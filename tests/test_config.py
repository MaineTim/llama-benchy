import pytest

from llama_benchy.config import BenchmarkConfig


def test_parse_extra_body_accepts_repeated_key_value_entries():
    extra = BenchmarkConfig._parse_extra_body([
        "min_tokens=1024",
        "ignore_eos=true",
        "temperature=0",
    ])

    assert extra == {
        "min_tokens": 1024,
        "ignore_eos": True,
        "temperature": 0,
    }


def test_parse_extra_body_accepts_aiperf_style_comma_separated_entries():
    extra = BenchmarkConfig._parse_extra_body([
        "max_tokens:1024,min_tokens:1024,ignore_eos:true",
    ])

    assert extra == {
        "max_tokens": 1024,
        "min_tokens": 1024,
        "ignore_eos": True,
    }


def test_parse_extra_body_rejects_entries_without_separator():
    with pytest.raises(ValueError):
        BenchmarkConfig._parse_extra_body(["ignore_eos"])


def _from_args(args):
    import sys

    old_argv = sys.argv
    sys.argv = ["llama-benchy"] + args
    try:
        return BenchmarkConfig.from_args()
    finally:
        sys.argv = old_argv


BASE_ARGS = ["--base-url", "http://localhost:1", "--model", "foo/bar"]


def test_prompt_offset_negative_rejected():
    with pytest.raises(SystemExit):
        _from_args(BASE_ARGS + ["--prompt-offset", "-5"])


def test_prompt_length_zero_rejected():
    with pytest.raises(SystemExit):
        _from_args(BASE_ARGS + ["--prompt-length", "0"])


def test_prompt_path_implies_fixed_prompt():
    cfg = _from_args(BASE_ARGS + ["--prompt-path", "/tmp/some-prompt.json"])
    assert cfg.fixed_prompt is True
    assert cfg.prompt_path == "/tmp/some-prompt.json"


def test_prompt_offset_implies_fixed_prompt():
    cfg = _from_args(BASE_ARGS + ["--prompt-offset", "42"])
    assert cfg.fixed_prompt is True


def test_new_fixed_prompt_implies_fixed_prompt():
    cfg = _from_args(BASE_ARGS + ["--new-fixed-prompt"])
    assert cfg.fixed_prompt is True
    assert cfg.new_fixed_prompt is True
