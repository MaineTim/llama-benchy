# llama-benchy (MaineTim fork)
Additions coded by Qwen3.8-27B under supervision of fork maintainer.

A fork of [eugr/llama-benchy](https://github.com/eugr/llama-benchy) that adds
**cross-session reproducible prompts** and a faster warmup for large test
shapes, so benchmark results can be compared across sessions, machines, and
backends (e.g. llama.cpp vs vLLM serving the same model).

Everything else — request flow, measurement, metrics, the progress stream —
is upstream code. The changes here are additive options; runs without the new
flags behave exactly like upstream.

## Base point

- Upstream: `eugr/llama-benchy`, commit `e9be344` (the upstream `main` tip as
  of the fork; one commit after the `v0.4.0` release tag).
- Fork commits on top (see `git log e9be344..HEAD`):

| Commit | What |
|--------|------|
| `2dfdab8` | First cut of `--fixed-prompt` (single pinned slice) |
| `6f173dc` | `--short-warmup` option |
| `02aa642` | `--fixed-prompt` rework: per-shape slice family (v2 format) |
| `c226fd6` | Catch-up: report/progress metadata, token-mismatch warning |
| `039f9eb` | Tests: `tests/test_promptfile.py`, `tests/test_fixed_matrix.py` |
| `ed5e5fe` | README documentation of the new behavior |

## What was added

### `--fixed-prompt` — reproducible prompts across sessions

By default llama-benchy samples a fresh random slice of the book text for
every request (good for avoiding prefix-cache hits, but it means two runs —
even of the same model on the same server — send *different* text, so their
numbers aren't directly comparable). `--fixed-prompt` pins the exact text:

- Each test shape (`pp` × `depth`) gets its **own** pinned character slice —
  a `pp16384` test never sends the text pinned for `pp2048`, which would
  silently inflate its throughput.
- Slices are cut by **character offsets, not tokens**, so the exact same text
  is sent regardless of tokenizer — which is what makes cross-model /
  cross-backend comparisons apples-to-apples.
- Slices are persisted to JSON files (`ctx<depth>_pp<pp>.json`) in a
  *family directory*; later sessions (any machine/user sharing the directory)
  read the files as-is, so identical content is guaranteed by construction.
  Default location: `~/.cache/llama-benchy/prompts/<source-hash>/`.
- Related flags: `--prompt-path` (choose/share a family), `--prompt-offset`,
  `--prompt-length`, `--prompt-context-length`, `--prompt-chars-per-token`,
  `--new-fixed-prompt` (re-cut with `.bak` backup). Conflicts with
  `--no-cache` (that one deliberately randomizes prompts).
- Full user docs: see the [Reproducible fixed prompts](README.md#reproducible-fixed-prompts)
  section in the main README.

Implementation lives in the new module `src/llama_benchy/promptfile.py`
(`PromptStore`, v2 file format), with wiring in `config.py`, `prompts.py`,
`runner.py`, `corpus.py`, and `__main__.py`.

#### Provenance: how to verify two runs used the same prompts

- Every run prints the source-text SHA-256 and, per shape, the slice's
  SHA-256, character count, and measured token count.
- JSON reports record `metadata.source_text_sha256` plus per-shape digests in
  `metadata.prompt_slices` (aggregate: `metadata.prompt_sha256`).
- `--emit-progress` streams the same digests in the `header` and
  `bench_complete` events.
- Two result sets used identical prompts iff the per-shape digests and the
  source-text hash match. (`prompt_tokens` legitimately differs across
  tokenizers — it's metadata, not identity.)
- New warning: if the *server-reported* prompt token count is far off the
  expected test size (e.g. the server truncated the prompt — context window
  too small), the run warns that the flagged test's prefill numbers should be
  treated as invalid.

### `--short-warmup`

When `--pp > 2048` or `--tg > 512`, warmup passes run at a reduced size
(pp ≤ 2048, tg ≤ 512) instead of the full size, which saves minutes on big
shapes. Measured runs always use the requested sizes; only discarded warmups
are capped.

### Working scripts (not yet committed)

`scripts/bench_matrix*.sh` — matrix-bench harnesses that run the same
`--pp 2048 16384 --depth 0 32768` matrix with `--fixed-prompt` across
backends (`bench_matrix_llama.sh` for a llama.cpp server,
`bench_matrix_vllm.sh` for vLLM, `bench_matrix.sh` for any OpenAI-compatible
endpoint), saving one result file per model under `results/`.

## What was *not* changed

- No changes to `client.py`, the request/sampling logic, or the metric
  calculations. The only touch to existing behavior in `results.py` is the
  new server-usage mismatch warning (above).
- The progress-stream schema changes are additive (new optional fields), so
  existing consumers keep working.
- The report JSON schema (`schemas/benchmark_report_schema.json`) gained new
  optional metadata fields only.

Rebasing onto upstream should stay cheap: the fork diff touches
`config.py`, `runner.py`, `prompts.py`, `results.py`, `corpus.py`,
`progress.py`, `__main__.py`, docs/schema, plus the new `promptfile.py` and
two test files.

## Versioning note

The version comes from **hatch-vcs** (git tags), baked into
`src/llama_benchy/_version.py` at build time. Forks do **not** inherit
upstream tags, so a fresh clone of this repo reports `0.0.0` until the tags
are fetched:

```bash
git remote add upstream https://github.com/eugr/llama-benchy.git
git fetch upstream --tags
```

After that, versions look like `0.4.1.dev<N>+g<hash>` (N commits after
`v0.4.0`). The stamp is frozen at install time, so **reinstall after
committing** (`uv sync --reinstall-package llama-benchy`) to get a fresh
version; a dirty tree adds a `.d<YYYYMMDD>` suffix. Check the running build
with `llama-benchy --version` — every run and JSON report carries the same
stamp, which is the provenance record for saved results.
