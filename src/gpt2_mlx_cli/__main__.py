"""Command line entry points for the terminal app and reproducible inference."""

from __future__ import annotations

import argparse
import json
import sys
import threading
from dataclasses import asdict
from pathlib import Path


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gpt2-mlx-cli", description="GPT-2 774M on MLX")
    commands = parser.add_subparsers(dest="command")
    tui = commands.add_parser("chat", help="Open the completion TUI (default)", allow_abbrev=False)
    tui.add_argument("--precision", choices=("fp16", "int8"), default="fp16")
    tui.add_argument("--model-dir", type=Path, help="Use an existing HF checkpoint directory")
    download = commands.add_parser("download", help="Download the pinned GPT-2 checkpoint")
    download.add_argument("--local-dir", type=Path)
    generate = commands.add_parser("generate", help="Generate a raw text continuation")
    generate.add_argument("prompt", nargs="?", default="Once upon a time")
    generate.add_argument("--precision", choices=("fp16", "int8"), default="fp16")
    generate.add_argument("--model-dir", type=Path)
    generate.add_argument("--max-tokens", type=int, default=128)
    generate.add_argument("--temperature", type=float, default=0.8)
    generate.add_argument("--top-p", type=float, default=0.95)
    generate.add_argument("--top-k", type=int, default=40)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument(
        "--json", action="store_true", help="Output generated text and metrics as JSON"
    )
    commands.add_parser("doctor", help="Check Python, MLX GPU execution and cached weights")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        if args.command in (None, "chat"):
            from gpt2_mlx_cli.engine import Engine
            from gpt2_mlx_cli.tui import GPT2App

            precision = getattr(args, "precision", "fp16")
            GPT2App(
                engine=Engine(precision, getattr(args, "model_dir", None)),
                precision=precision,
            ).run()
        elif args.command == "download":
            from huggingface_hub import snapshot_download

            from gpt2_mlx_cli.engine import MODEL_FILES, MODEL_ID, MODEL_REVISION

            print(
                snapshot_download(
                    MODEL_ID,
                    revision=MODEL_REVISION,
                    allow_patterns=MODEL_FILES,
                    local_dir=args.local_dir,
                    max_workers=2,
                )
            )
        elif args.command == "doctor":
            import mlx.core as mx
            from huggingface_hub import snapshot_download

            from gpt2_mlx_cli.engine import MODEL_FILES, MODEL_ID, MODEL_REVISION

            a = mx.ones((64, 64), dtype=mx.float16)
            result = a @ a
            mx.eval(result)
            print(f"Python: {sys.version.split()[0]}")
            print(f"MLX device: {mx.device_info()['device_name']}")
            print(f"GPU matmul: {'OK' if result[0, 0].item() == 64 else 'FAILED'}")
            try:
                path = Path(
                    snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
                )
                cached = all((path / name).is_file() for name in MODEL_FILES)
            except (OSError, ValueError):
                cached = False
            print(f"Checkpoint: {'cached' if cached else 'run: uv run gpt2-mlx-cli download'}")
        else:
            from gpt2_mlx_cli.engine import Engine, GenerationSettings

            settings = GenerationSettings(
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=args.max_tokens,
                seed=args.seed,
            )
            engine = Engine(args.precision, args.model_dir)
            engine.load(lambda message: print(message, file=sys.stderr))
            stop = threading.Event()
            previous = ""
            final = None
            for event in engine.generate(engine.tokenizer.encode(args.prompt), settings, stop):
                if not args.json:
                    print(event.text[len(previous) :], end="", flush=True)
                previous = event.text
                final = event
            if args.json:
                print(json.dumps(asdict(final), ensure_ascii=False, indent=2))
            else:
                print()
                print(
                    f"{final.token_count} tokens · {final.tokens_per_second:.1f} tok/s · "
                    f"{final.peak_memory_bytes / 1024**3:.2f} GiB MLX peak · "
                    f"{final.finish_reason}",
                    file=sys.stderr,
                )
        return 0
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, RuntimeError) as error:
        print(f"gpt2-mlx-cli: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
