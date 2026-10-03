"""
Main entry point for the llama-benchy CLI.
"""

import asyncio
import datetime
import os
import sys
from . import __version__
from .config import BenchmarkConfig
from .corpus import TokenizedCorpus
from .promptfile import check_explicit_params, load_or_create, sha256_text
from .prompts import PromptGenerator
from .client import LLMClient
from .runner import BenchmarkRunner
from .progress import ProgressEmitter

DEFAULT_PROMPT_LENGTH_PER_TOKEN = 4  # ~4 chars per token for English
DEFAULT_CONTEXT_LENGTH = 4096

async def main_async():
    # 1. Parse Configuration
    config = BenchmarkConfig.from_args()

    # 1b. If JSONL is going to stdout, route llama-benchy's status prints to
    # stderr so they don't corrupt the JSONL stream a consumer is parsing.
    if config.emit_progress == "-":
        sys.stdout = sys.stderr

    # 2. Print Header
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"llama-benchy ({__version__})")
    print(f"Date: {current_time}")
    print(f"Benchmarking model: {config.model} at {config.base_url}")
    print(f"Concurrency levels: {config.concurrency_levels}")

    if config.fixed_prompt:
        print(
            "Note: fixed prompt in use — the same prompt text is sent for every run and "
            "every session. If the server has prompt caching enabled, prefill (pp) numbers "
            "may be inflated by cache hits. Make sure prompt caching is disabled on the "
            "server, otherwise ignore the prefill numbers reported below."
        )

    # 3. Prepare Data
    corpus = TokenizedCorpus(config.book_url, config.tokenizer, config.model)
    print(
        f"Source text: {config.book_url} "
        f"(sha256: {corpus.get_text_sha256()}, {corpus.get_text_chars()} chars)"
    )
    print(f"Total tokens available in text corpus: {len(corpus)}")

    # 3b. Resolve the fixed prompt file (when --fixed-prompt family is in use)
    fixed_prompt = None
    prompt_path = None
    if config.fixed_prompt:
        offset = config.prompt_offset if config.prompt_offset is not None else 0
        length = (
            config.prompt_length
            if config.prompt_length is not None
            else DEFAULT_PROMPT_LENGTH_PER_TOKEN * config.pp_counts[0]
        )
        context_length = (
            config.prompt_context_length
            if config.prompt_context_length is not None
            else DEFAULT_CONTEXT_LENGTH
        )
        if config.prompt_path:
            prompt_path = config.prompt_path
        else:
            default_dir = os.path.join(
                os.path.expanduser("~"), ".cache", "llama-benchy", "prompts"
            )
            prompt_path = os.path.join(
                default_dir, f"prompt-{corpus.get_text_sha256()[:16]}.json"
            )

        result = load_or_create(
            prompt_path,
            book_url=config.book_url,
            source_sha256=corpus.get_text_sha256(),
            text=corpus.get_text(),
            offset=offset,
            context_length=context_length,
            prompt_length=length,
            new=config.new_fixed_prompt,
        )
        fixed_prompt = result.prompt

        if result.status == "loaded":
            mismatch = check_explicit_params(
                fixed_prompt,
                offset=config.prompt_offset,
                prompt_length=config.prompt_length,
                context_length=config.prompt_context_length,
            )
            if mismatch:
                print(
                    f"Error: {mismatch}. The prompt file is the source of truth; use a "
                    "different --prompt-path for a new fixed prompt family, or "
                    "--new-fixed-prompt to overwrite this one (it is backed up to <path>.bak)."
                )
                sys.exit(1)
            print(f"Fixed prompt: {prompt_path} (loaded)")
        elif result.status == "created":
            print(f"Fixed prompt: {prompt_path} (created — later sessions will reuse this file)")
        else:  # regenerated
            print(
                f"Note: --new-fixed-prompt — regenerated prompt file at {prompt_path} "
                f"(previous prompt sha256: {result.previous_sha256}, "
                f"new prompt sha256: {sha256_text(fixed_prompt.prompt)}). "
                f"Old file backed up to {prompt_path}.bak. "
                "Results from earlier sessions are NOT comparable to this one."
            )

        if fixed_prompt.source_sha256 != corpus.get_text_sha256():
            print(
                f"Warning: prompt file was cut from a different source text "
                f"(file sha256: {fixed_prompt.source_sha256}, "
                f"current text sha256: {corpus.get_text_sha256()}). "
                "The file content is used as-is."
            )

        tokenizer = corpus.get_tokenizer()
        prompt_tokens = len(tokenizer.encode(fixed_prompt.prompt, add_special_tokens=False))
        print(
            f"  prompt sha256: {sha256_text(fixed_prompt.prompt)} "
            f"({fixed_prompt.prompt_length} chars / {prompt_tokens} tokens, "
            f"offset {fixed_prompt.offset})"
        )
        if fixed_prompt.context_length > 0:
            print(f"  context: {fixed_prompt.context_length} chars (sent when --depth > 0)")

        if config.adapt_prompt:
            print(
                "Tip: with a fixed prompt the sent text never changes; --adapt-prompt only "
                "adjusts the expected token counts used in the metrics. Consider "
                "--no-adapt-prompt for cleaner comparisons."
            )

    # 4. Initialize Components
    prompt_gen = PromptGenerator(corpus, fixed_prompt=fixed_prompt)
    client = LLMClient(
        config.base_url,
        config.api_key,
        config.served_model_name,
        config.extra_body,
        config.exact_tg,
    )

    prompt_metadata = {
        "source_text_sha256": corpus.get_text_sha256(),
        "source_text_chars": corpus.get_text_chars(),
    }
    if fixed_prompt is not None:
        prompt_metadata.update(
            {
                "prompt_file": prompt_path,
                "prompt_sha256": sha256_text(fixed_prompt.prompt),
                "prompt_chars": len(fixed_prompt.prompt),
                "prompt_tokens": len(
                    corpus.get_tokenizer().encode(fixed_prompt.prompt, add_special_tokens=False)
                ),
            }
        )

    progress = None
    if config.emit_progress:
        progress = ProgressEmitter(
            config.emit_progress,
            llama_benchy_version=__version__,
            source_text_sha256=prompt_metadata.get("source_text_sha256"),
            prompt_sha256=prompt_metadata.get("prompt_sha256"),
            prompt_tokens=prompt_metadata.get("prompt_tokens"),
        )
    runner = BenchmarkRunner(
        config, client, prompt_gen, progress=progress, prompt_metadata=prompt_metadata
    )

    # 5. Run Benchmark Suite
    status = "ok"
    try:
        await runner.run_suite()
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    except BaseException:
        status = "error"
        raise
    finally:
        if progress is not None:
            try:
                progress.bench_complete(status=status)
            finally:
                progress.close()

    print(f"\nllama-benchy ({__version__})")
    print(f"date: {current_time} | latency mode: {config.latency_mode}")

def main():
    """Entry point for the CLI command."""
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        sys.exit(1)

if __name__ == "__main__":
    main()
