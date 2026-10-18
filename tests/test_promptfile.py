import json
import os

import pytest

from llama_benchy.promptfile import (
    SCHEMA_VERSION,
    PromptMismatchError,
    PromptStore,
    sha256_text,
    slice_filename,
    slice_text,
)

TEXT = ("The quick brown fox jumps over the lazy dog. " * 400)[:-1]


class _CharTokenizer:
    """Extremely simple tokenizer: one token per char."""

    def encode(self, text, add_special_tokens=False):
        return list(range(len(text)))


def _make_store(tmp_path, **overrides):
    kwargs = dict(
        book_url="https://example.com/book.txt",
        source_sha256=sha256_text(TEXT),
        text=TEXT,
        tokenizer=_CharTokenizer(),
        tokenizer_name="fake/tokenizer",
        offset=10,
        chars_per_token=1,  # one char per token keeps the math readable
        new=False,
    )
    kwargs.update(overrides)
    family_dir = str(tmp_path / "family")
    return PromptStore(family_dir, **kwargs)


def _read_manifest(path):
    with open(path) as f:
        return json.load(f)


def test_slice_text_contiguous():
    ctx, pr = slice_text("abcdefghij", 2, 3, 4)
    assert ctx == "cde"
    assert pr == "fghi"


def test_slice_text_zero_context():
    ctx, pr = slice_text("abcdefghij", 2, 0, 4)
    assert ctx == ""
    assert pr == "cdef"


@pytest.mark.parametrize(
    "offset,ctx_len,pr_len",
    [(-1, 5, 5), (0, -1, 5), (0, 5, 0), (0, 5, 500)],
)
def test_slice_text_rejects_bad_bounds(offset, ctx_len, pr_len):
    with pytest.raises(ValueError):
        slice_text("short text", offset, ctx_len, pr_len)


def test_slice_filename_encodes_shape():
    assert slice_filename(16384, 32768) == "ctx32768_pp16384.json"


def test_store_creates_one_file_per_shape(tmp_path):
    store = _make_store(tmp_path)
    small = store.get(8, 0)
    large = store.get(16, 0)

    assert small.status == "created"
    # The whole point of the fix: distinct shapes must get distinct text,
    # sized from their own nominal token counts.
    assert len(small.prompt.prompt) == 8
    assert len(large.prompt.prompt) == 16
    assert small.prompt.prompt != large.prompt.prompt
    # Slices are one contiguous region, so the short prompt prefixes the long one.
    assert large.prompt.prompt.startswith(small.prompt.prompt)
    assert sorted(os.listdir(store.family_dir)) == [
        "ctx0_pp16.json",
        "ctx0_pp8.json",
    ]


def test_store_slices_scale_with_depth(tmp_path):
    store = _make_store(tmp_path)
    shallow = store.get(8, 4)
    deep = store.get(8, 32)
    assert len(shallow.prompt.context) == 4
    assert len(deep.prompt.context) == 32
    # Context and prompt are one contiguous region of the source text.
    assert TEXT[10:10 + 32 + 8] == deep.prompt.context + deep.prompt.prompt
    assert deep.prompt.context_length == 32 and deep.prompt.prompt_length == 8


def test_store_depth_zero_has_empty_context(tmp_path):
    store = _make_store(tmp_path)
    result = store.get(8, 0)
    assert result.prompt.context == ""
    assert result.prompt.context_length == 0


def test_manifest_v2_records_shape_and_tokens(tmp_path):
    store = _make_store(tmp_path)
    store.get(8, 4)
    data = _read_manifest(os.path.join(store.family_dir, "ctx4_pp8.json"))
    assert data["schema"] == SCHEMA_VERSION
    assert data["pp"] == 8
    assert data["depth"] == 4
    assert data["prompt_tokens"] == 8  # one char per token
    assert data["tokenizer"] == "fake/tokenizer"
    assert data["offset"] == 10
    assert data["source_sha256"] == sha256_text(TEXT)


def test_store_reads_existing_file_as_is(tmp_path):
    store = _make_store(tmp_path)
    store.get(8, 0)

    # New store, same dir, different params: the file wins.
    store2 = _make_store(tmp_path, offset=500, chars_per_token=2)
    result = store2.get(8, 0)
    assert result.status == "loaded"
    assert len(result.prompt.prompt) == 8
    assert result.prompt.offset == 10


def test_store_new_regenerates_with_backup(tmp_path):
    store = _make_store(tmp_path)
    old = store.get(8, 0).prompt
    old_sha = sha256_text(old.prompt)

    store2 = _make_store(tmp_path, offset=200, new=True)
    result = store2.get(8, 0)
    assert result.status == "regenerated"
    assert result.previous_sha256 == old_sha
    assert result.prompt.offset == 200
    assert result.prompt.prompt == TEXT[200:208]

    backup = os.path.join(store2.family_dir, "ctx0_pp8.json.bak")
    assert sha256_text(_read_manifest(backup)["prompt"]) == old_sha


def test_store_new_regenerates_only_after_file_exists(tmp_path):
    store = _make_store(tmp_path, new=True)
    result = store.get(8, 0)
    assert result.status == "created"
    assert result.previous_sha256 is None


def test_store_caches_per_shape(tmp_path):
    store = _make_store(tmp_path)
    a = store.get(8, 0)
    b = store.get(8, 0)
    assert a is b
    assert store.shapes() == [(8, 0)]


def test_explicit_offset_mismatch_raises(tmp_path):
    store = _make_store(tmp_path)
    store.get(8, 0)

    store2 = _make_store(tmp_path, offset=99, explicit_offset=True)
    with pytest.raises(PromptMismatchError, match="offset"):
        store2.get(8, 0)


def test_explicit_prompt_length_mismatch_raises(tmp_path):
    store = _make_store(tmp_path)
    store.get(8, 0)

    store2 = _make_store(tmp_path, prompt_length=123, explicit_prompt_length=True)
    with pytest.raises(PromptMismatchError, match="prompt-length"):
        store2.get(8, 0)


def test_explicit_context_length_mismatch_raises(tmp_path):
    store = _make_store(tmp_path)
    store.get(8, 4)

    store2 = _make_store(tmp_path, context_length=123, explicit_context_length=True)
    with pytest.raises(PromptMismatchError, match="prompt-context-length"):
        store2.get(8, 4)


def test_source_sha_mismatch_warns(tmp_path):
    store = _make_store(tmp_path)
    store.get(8, 0)

    store2 = _make_store(tmp_path, source_sha256="0" * 64)
    result = store2.get(8, 0)
    assert result.status == "loaded"
    assert result.warning is not None
    assert "different source text" in result.warning


def test_token_count_deviation_warns(tmp_path):
    store = _make_store(tmp_path)
    store.get(8, 0)
    # Load the slice but claim it is for a pp=100 shape by rewriting pp in
    # the manifest (simulates a char/token ratio far from nominal).
    path = os.path.join(store.family_dir, "ctx0_pp8.json")
    data = _read_manifest(path)
    data["pp"] = 100
    with open(path, "w") as f:
        json.dump(data, f)

    store2 = _make_store(tmp_path)
    result = store2.get(8, 0)
    assert result.warning is not None
    assert "tokens" in result.warning and "100" in result.warning


def test_rejects_unknown_schema(tmp_path):
    store = _make_store(tmp_path)
    os.makedirs(store.family_dir, exist_ok=True)
    path = os.path.join(store.family_dir, slice_filename(8, 0))
    with open(path, "w") as f:
        json.dump({"schema": "something-else.v9", "prompt": "x"}, f)
    with pytest.raises(ValueError, match="unknown schema"):
        store.get(8, 0)


def _write_v1(path, prompt_text, context_text=""):
    # v1 files carry no schema-versioned shape fields.
    manifest = {
        "schema": "llama-benchy-prompt.v1",
        "book_url": "https://example.com/book.txt",
        "source_sha256": sha256_text(TEXT),
        "offset": 10,
        "context_length": len(context_text),
        "prompt_length": len(prompt_text),
        "context": context_text,
        "prompt": prompt_text,
        "created_at": "2025-01-01T00:00:00Z",
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f)


def test_legacy_file_serves_matching_shape(tmp_path):
    legacy = str(tmp_path / "legacy.json")
    _write_v1(legacy, TEXT[10:18])  # 8 chars == 8 tokens

    store = _make_store(tmp_path, legacy_path=legacy)
    match = store.get(8, 0)
    assert match.status == "legacy"
    assert match.warning and "legacy" in match.warning
    assert match.prompt.prompt == TEXT[10:18]
    # Only written files count: the legacy file itself is never touched.
    assert not os.path.exists(os.path.join(store.family_dir, slice_filename(8, 0)))

    # A shape that does not match the legacy slice gets its own v2 file.
    other = store.get(64, 0)
    assert other.status == "created"
    assert len(other.prompt.prompt) == 64


def test_legacy_context_presence_must_match_depth(tmp_path):
    legacy = str(tmp_path / "legacy.json")
    _write_v1(legacy, TEXT[10:18], context_text=TEXT[0:10])

    store = _make_store(tmp_path, legacy_path=legacy)
    # Legacy slice has a context; a depth=0 shape does not want one.
    assert store.get(8, 0).status == "created"


def test_legacy_bypassed_with_new_flag(tmp_path):
    legacy = str(tmp_path / "legacy.json")
    _write_v1(legacy, TEXT[10:18])

    store = _make_store(tmp_path, legacy_path=legacy, offset=500, new=True)
    result = store.get(8, 0)
    assert result.status == "created"
    assert result.prompt.offset == 500


def test_legacy_unreadable_is_ignored(tmp_path):
    legacy = str(tmp_path / "legacy.json")
    os.makedirs(tmp_path, exist_ok=True)
    with open(legacy, "w") as f:
        f.write("not json")

    store = _make_store(tmp_path, legacy_path=legacy)
    assert store.get(8, 0).status == "created"
