"""
Main entry point for the llama-benchy CLI.
"""

import asyncio
import datetime
import hashlib
import json
import os
import sys
from . import __version__
from .config import BenchmarkConfig
from .corpus import TokenizedCorpus
from .promptfile import PromptMismatchError, PromptStore, sha256_text
from .prompts import PromptGenerator
from .client import LLMClient
from .runner import BenchmarkRunner
from .progress import ProgressEmitter

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
            "Note: fixed prompt in use — the pinned prompt text for each pp/depth shape is "
            "sent for every run and every session. If the server has prompt caching enabled, prefill (pp) numbers "
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

    # 3b. Resolve the fixed prompt family (when --fixed-prompt is in use):
    # one pinned slice per (pp, depth) test shape.
    prompt_store = None
    prompt_path = None
    if config.fixed_prompt:
        tokenizer = corpus.get_tokenizer()
        tokenizer_name = config.tokenizer or config.model

        prompts_root = os.path.join(
            os.path.expanduser("~"), ".cache", "llama-benchy", "prompts"
        )
        sha16 = corpus.get_text_sha256()[:16]
        legacy_path = None
        if config.prompt_path:
            configured = os.path.abspath(config.prompt_path)
            if os.path.isfile(configured):
                # Pre-v2 usage: --prompt-path pointed at a single slice file.
                # Keep serving matching shapes from it; new shapes go into a
                # sibling family directory.
                legacy_path = configured
                family_dir = configured + ".d"
            else:
                family_dir = configured
        else:
            family_dir = os.path.join(prompts_root, sha16)
            candidate = os.path.join(prompts_root, f"prompt-{sha16}.json")
            if os.path.exists(candidate):
                legacy_path = candidate
        prompt_path = family_dir

        prompt_store = PromptStore(
            family_dir,
            book_url=config.book_url,
            source_sha256=corpus.get_text_sha256(),
            text=corpus.get_text(),
            tokenizer=tokenizer,
            tokenizer_name=tokenizer_name,
            offset=config.prompt_offset if config.prompt_offset is not None else 0,
            chars_per_token=config.prompt_chars_per_token,
            prompt_length=config.prompt_length,
            context_length=config.prompt_context_length,
            new=config.new_fixed_prompt,
            legacy_path=legacy_path,
            explicit_offset=config.prompt_offset is not None,
            explicit_prompt_length=config.prompt_length is not None,
            explicit_context_length=config.prompt_context_length is not None,
        )

        print(f"Fixed prompt family: {family_dir} (one pinned slice per pp/depth shape)")
        shapes = [
            (pp, depth)
            for depth in config.depths
            for pp in config.pp_counts
        ]
        try:
            for pp, depth in shapes:
                prompt_store.get(pp, depth)
        except PromptMismatchError as e:
            print(
                f"Error: {e}. The pinned slice file is the source of truth; use a "
                "different --prompt-path for a new fixed prompt family, or "
                "--new-fixed-prompt to regenerate it (each file is backed up to <file>.bak)."
            )
            sys.exit(1)
        except ValueError as e:
            print(f"Error preparing fixed prompt family: {e}")
            sys.exit(1)

        for pp, depth in shapes:
            result = prompt_store.get(pp, depth)
            fp = result.prompt
            tokens = fp.prompt_tokens
            if tokens is None:
                tokens = len(tokenizer.encode(fp.prompt, add_special_tokens=False))
            ctx_info = f" + ctx {len(fp.context)} chars" if depth > 0 and fp.context else ""
            print(
                f"  slice pp{pp}@d{depth} [{result.status}]: sha256 {sha256_text(fp.prompt)[:16]} "
                f"({len(fp.prompt)} chars / {tokens} tokens{ctx_info}, offset {fp.offset})"
            )
            if result.status == "regenerated":
                print(
                    f"    Note: --new-fixed-prompt — old slice for this shape had sha256 "
                    f"{result.previous_sha256}; the file was backed up to <file>.bak. "
                    "Results from earlier sessions are NOT comparable to this one."
                )
            if result.warning:
                for line in result.warning.split("\n"):
                    print(f"    Warning: {line}")

        if config.adapt_prompt:
            print(
                "Tip: with a fixed prompt the sent text never changes; --adapt-prompt only "
                "adjusts the expected token counts used in the metrics. Consider "
                "--no-adapt-prompt for cleaner comparisons."
            )

    # 4. Initialize Components
    prompt_gen = PromptGenerator(corpus, prompt_store=prompt_store)
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
    if prompt_store is not None:
        slices_meta = {}
        for pp, depth in prompt_store.shapes():
            fp = prompt_store.get(pp, depth).prompt
            slices_meta[f"pp{pp}@d{depth}"] = {
                "prompt_sha256": sha256_text(fp.prompt),
                "prompt_chars": len(fp.prompt),
                "prompt_tokens": fp.prompt_tokens
                if fp.prompt_tokens is not None
                else len(corpus.get_tokenizer().encode(fp.prompt, add_special_tokens=False)),
                "context_chars": len(fp.context),
            }
        family_digest = hashlib.sha256(
            json.dumps(slices_meta, sort_keys=True).encode("utf-8")
        ).hexdigest()
        prompt_metadata.update(
            {
                "prompt_file": prompt_path,
                "prompt_sha256": family_digest,
                "prompt_chars": sum(s["prompt_chars"] for s in slices_meta.values()),
                "prompt_tokens": sum(s["prompt_tokens"] for s in slices_meta.values()),
                "prompt_slices": slices_meta,
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
