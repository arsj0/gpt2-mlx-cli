"""Local checkpoint loading and cancellable, streaming token generation."""

from __future__ import annotations

import codecs
import gc
import math
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer as RustTokenizer

from gpt2_mlx_cli.model import GPT2, load_checkpoint

MODEL_ID = "openai-community/gpt2-large"
MODEL_REVISION = "32b71b12589c2f8d625668d2335a01cac3249519"
MODEL_FILES = ["config.json", "tokenizer.json", "model.safetensors"]
EOS_TOKEN = 50256
CONTEXT_LENGTH = 1024


@dataclass(frozen=True)
class GenerationSettings:
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 40
    max_new_tokens: int = 128
    seed: int = 42

    def __post_init__(self) -> None:
        if not math.isfinite(self.temperature) or not 0 <= self.temperature <= 5:
            raise ValueError("Temperature must be between 0 and 5 (0 means greedy).")
        if not math.isfinite(self.top_p) or not 0 < self.top_p <= 1:
            raise ValueError("Top-p must be greater than 0 and at most 1.")
        if type(self.top_k) is not int or not 0 <= self.top_k <= 50257:
            raise ValueError("Top-k must be an integer from 0 to 50257 (0 disables it).")
        if type(self.max_new_tokens) is not int or not 1 <= self.max_new_tokens < CONTEXT_LENGTH:
            raise ValueError("Output length must be an integer from 1 to 1023.")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("Seed must be an integer from 0 to 4294967295.")


@dataclass(frozen=True)
class GenerationEvent:
    text: str = ""
    token_count: int = 0
    prompt_tokens: int = 0
    elapsed: float = 0
    tokens_per_second: float = 0
    finish_reason: str | None = None
    peak_memory_bytes: int = 0
    phase: str = "prefill"


class Tokenizer:
    """Expose plain token IDs and incremental, lossless UTF-8 decoding."""

    eos_token_id = EOS_TOKEN

    def __init__(self, path: Path):
        self.inner = RustTokenizer.from_file(str(path))
        byte_values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        visible = set(byte_values)
        codepoints = byte_values.copy()
        for byte in range(256):
            if byte not in visible:
                byte_values.append(byte)
                codepoints.append(256 + len(codepoints) - len(visible))
        self.byte_decoder = {
            chr(code): byte for byte, code in zip(byte_values, codepoints, strict=True)
        }

    def encode(self, text: str) -> list[int]:
        return self.inner.encode(text, add_special_tokens=False).ids

    def decode(self, tokens: list[int]) -> str:
        return self.inner.decode(tokens, skip_special_tokens=False)

    def token_bytes(self, token: int) -> bytes:
        piece = self.inner.id_to_token(token)
        if piece is None:
            raise ValueError(f"Unknown token ID: {token}")
        return bytes(self.byte_decoder[character] for character in piece)


class StopBuffer:
    """Withhold partial role markers so they never flash in streamed output."""

    def __init__(self, stops: tuple[str, ...] = ()):
        self.stops = tuple(stop for stop in stops if stop)
        self.text = ""
        self.stopped = False

    def update(self, text: str, *, final: bool = False) -> str:
        indices = [text.find(stop) for stop in self.stops if stop in text]
        if indices:
            self.stopped = True
            self.text = text[: min(indices)]
            return self.text
        withheld = 0
        if not final:
            for stop in self.stops:
                for size in range(1, min(len(stop), len(text) + 1)):
                    if text.endswith(stop[:size]):
                        withheld = max(withheld, size)
        self.text = text[:-withheld] if withheld else text
        return self.text


def sample(logits: mx.array, settings: GenerationSettings, key: mx.array) -> mx.array:
    """Sample in FP32, applying top-k before nucleus filtering."""
    logits = logits.astype(mx.float32)
    if settings.temperature == 0:
        return mx.argmax(logits)
    logits = logits / settings.temperature
    indices = None
    if settings.top_k and settings.top_k < logits.size:
        indices = mx.argpartition(-logits, kth=settings.top_k - 1)[: settings.top_k]
        logits = logits[indices]
    if settings.top_p < 1:
        order = mx.argsort(-logits)
        sorted_logits = logits[order]
        probabilities = mx.softmax(sorted_logits)
        keep = mx.cumsum(probabilities) - probabilities < settings.top_p
        sorted_logits = mx.where(keep, sorted_logits, -float("inf"))
        choice = order[mx.random.categorical(sorted_logits, key=key)]
    else:
        choice = mx.random.categorical(logits, key=key)
    return indices[choice] if indices is not None else choice


class Engine:
    """One loaded model; callers serialize all operations on a single worker."""

    def __init__(self, precision: str = "fp16", model_dir: Path | None = None):
        if precision not in ("fp16", "int8"):
            raise ValueError("Choose fp16 or int8.")
        self.precision = precision
        self.model_dir = Path(model_dir) if model_dir else None
        self.model: GPT2 | None = None
        self.tokenizer: Tokenizer | None = None

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def load(
        self, progress: Callable[[str], None] | None = None, stop: threading.Event | None = None
    ) -> None:
        if self.loaded:
            return
        stop = stop or threading.Event()
        if stop.is_set():
            raise InterruptedError("Model loading cancelled.")
        # A bounded allocator cache prevents idle inference buffers retaining gigabytes.
        mx.set_cache_limit(128 * 1024 * 1024)
        if self.model_dir is None:
            try:
                path = snapshot_download(
                    MODEL_ID,
                    revision=MODEL_REVISION,
                    allow_patterns=MODEL_FILES,
                    local_files_only=True,
                )
                if not all((Path(path) / filename).is_file() for filename in MODEL_FILES):
                    raise FileNotFoundError("Checkpoint is not fully cached.")
            except (OSError, ValueError):
                if progress:
                    progress("Downloading GPT-2 774M · first run needs about 3.25 GB")
                self._download(stop)
                path = snapshot_download(
                    MODEL_ID,
                    revision=MODEL_REVISION,
                    allow_patterns=MODEL_FILES,
                    local_files_only=True,
                )
            self.model_dir = Path(path)
        if progress:
            progress("Loading GPT-2 tokenizer")
        try:
            self.tokenizer = Tokenizer(self.model_dir / "tokenizer.json")
            self.model = load_checkpoint(self.model_dir, self.precision, progress, stop)
            if stop.is_set():
                raise InterruptedError("Model loading cancelled.")
        except BaseException:
            self.unload()
            raise
        mx.clear_cache()

    @staticmethod
    def _download(stop: threading.Event) -> None:
        # HF's transfer threads cannot be interrupted reliably. Isolate only the download.
        with tempfile.TemporaryFile() as log:
            process = subprocess.Popen(
                [sys.executable, "-m", "gpt2_mlx_cli", "download"],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            try:
                while process.poll() is None:
                    if stop.wait(0.1):
                        raise InterruptedError("Model download cancelled; cached data is retained.")
                if process.returncode != 0:
                    log.seek(0)
                    detail = log.read().decode("utf-8", errors="replace")[-2000:]
                    raise RuntimeError(f"Model download failed. Retry when connected.\n{detail}")
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()

    def unload(self) -> None:
        self.model = None
        self.tokenizer = None
        gc.collect()
        mx.clear_cache()

    def set_precision(
        self,
        value: str,
        progress: Callable[[str], None] | None = None,
        stop: threading.Event | None = None,
    ) -> None:
        if value not in ("fp16", "int8"):
            raise ValueError("Choose fp16 or int8.")
        if value == self.precision and self.loaded:
            return
        self.unload()
        self.precision = value
        self.load(progress, stop)

    def generate(
        self,
        prompt_ids: list[int],
        settings: GenerationSettings,
        stop: threading.Event | None = None,
        stop_strings: tuple[str, ...] = (),
    ) -> Iterator[GenerationEvent]:
        if not self.loaded or self.tokenizer is None:
            raise RuntimeError("Load the model before generating.")
        if not prompt_ids:
            prompt_ids = [EOS_TOKEN]
        if any(
            type(token) is not int or not 0 <= token < self.model.config.vocab_size
            for token in prompt_ids
        ):
            raise ValueError("Prompt contains an invalid token ID.")
        if len(prompt_ids) + settings.max_new_tokens > self.model.config.n_positions:
            raise ValueError("Prompt and requested output exceed the 1024-token context window.")
        stop = stop or threading.Event()
        cache = self.model.make_cache(len(prompt_ids) + settings.max_new_tokens)
        mx.reset_peak_memory()
        start = time.perf_counter()
        first_token_at = None
        generated = 0
        text = ""
        buffer = StopBuffer(stop_strings)
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        key = mx.random.key(settings.seed)

        def event(phase: str, reason: str | None = None) -> GenerationEvent:
            now = time.perf_counter()
            decode_elapsed = now - first_token_at if first_token_at else 0
            speed = max(0, generated - 1) / decode_elapsed if decode_elapsed > 0 else 0
            return GenerationEvent(
                buffer.text,
                generated,
                len(prompt_ids),
                now - start,
                speed,
                reason,
                mx.get_peak_memory(),
                phase,
            )

        try:
            yield event("prefill")
            if stop.is_set():
                yield event("done", "cancelled")
                return
            # Chunking bounds prefill activation memory and gives cancellation checkpoints.
            logits = None
            for offset in range(0, len(prompt_ids), 128):
                chunk = mx.array([prompt_ids[offset : offset + 128]], dtype=mx.int32)
                logits = self.model(chunk, cache)[0]
                mx.eval(logits, *[value for layer in cache for value in (layer.keys, layer.values)])
                if stop.is_set():
                    yield event("done", "cancelled")
                    return
            reason = "length"
            for index in range(settings.max_new_tokens):
                if stop.is_set():
                    reason = "cancelled"
                    break
                key, subkey = mx.random.split(key)
                token = int(sample(logits, settings, subkey).item())
                generated += 1
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                if token == EOS_TOKEN:
                    reason = "eos"
                    break
                text += decoder.decode(self.tokenizer.token_bytes(token))
                buffer.update(text)
                if buffer.stopped:
                    reason = "stop"
                    break
                yield event("decode")
                if index + 1 < settings.max_new_tokens and not stop.is_set():
                    logits = self.model(mx.array([[token]], dtype=mx.int32), cache)[0]
                    mx.eval(
                        logits, *[value for layer in cache for value in (layer.keys, layer.values)]
                    )
            if reason != "stop":
                # Interrupted streams may end mid-codepoint. Preserve complete characters.
                if reason == "eos":
                    text += decoder.decode(b"", final=True)
                buffer.update(text, final=True)
            yield event("done", reason)
        finally:
            cache.clear()
            mx.clear_cache()
