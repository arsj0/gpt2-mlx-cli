"""GPT-2 forward inference built directly from MLX operations."""

from __future__ import annotations

import json
import math
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
from safetensors import safe_open


@mx.compile
def gelu(x: mx.array) -> mx.array:
    """Fuse the original GPT-2 tanh approximation into a single elementwise kernel."""
    return 0.5 * x * (1 + mx.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 50257
    n_positions: int = 1024
    n_embd: int = 1280
    n_layer: int = 36
    n_head: int = 20
    layer_norm_epsilon: float = 1e-5

    @classmethod
    def from_file(cls, path: Path) -> ModelConfig:
        data = json.loads(path.read_text())
        if data.get("model_type") != "gpt2" or data.get("activation_function") != "gelu_new":
            raise ValueError("Expected a GPT-2 checkpoint with the original GELU activation.")
        return cls(**{name: data[name] for name in cls.__dataclass_fields__})


class KVCache:
    """Allocate only the requested context budget, then update it in place."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.offset = 0
        self.keys: mx.array | None = None
        self.values: mx.array | None = None

    def update(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        end = self.offset + keys.shape[2]
        if end > self.capacity:
            raise ValueError("KV cache capacity exceeded.")
        if self.keys is None:
            shape = (*keys.shape[:2], self.capacity, keys.shape[3])
            self.keys = mx.zeros(shape, dtype=keys.dtype)
            self.values = mx.zeros(shape, dtype=values.dtype)
        self.keys[:, :, self.offset : end, :] = keys
        self.values[:, :, self.offset : end, :] = values
        self.offset = end
        return self.keys[:, :, :end, :], self.values[:, :, :end, :]


class GPT2:
    """An inference-only decoder; the token embedding is also the output head."""

    def __init__(self, config: ModelConfig, weights: dict[str, mx.array]):
        self.config = config
        self.weights = weights
        if config.n_embd % config.n_head:
            raise ValueError("The embedding dimension must be divisible by the head count.")
        self._validate_weights()

    def _validate_weights(self) -> None:
        c = self.config
        expected = {
            "wte.weight": (c.vocab_size, c.n_embd),
            "wpe.weight": (c.n_positions, c.n_embd),
            "ln_f.weight": (c.n_embd,),
            "ln_f.bias": (c.n_embd,),
        }
        for i in range(c.n_layer):
            for norm in ("ln_1", "ln_2"):
                for kind in ("weight", "bias"):
                    expected[f"h.{i}.{norm}.{kind}"] = (c.n_embd,)
            for name, shape in {
                "attn.c_attn": (3 * c.n_embd, c.n_embd),
                "attn.c_proj": (c.n_embd, c.n_embd),
                "mlp.c_fc": (4 * c.n_embd, c.n_embd),
                "mlp.c_proj": (c.n_embd, 4 * c.n_embd),
            }.items():
                prefix = f"h.{i}.{name}"
                expected[f"{prefix}.bias"] = (shape[0],)
                if f"{prefix}.scales" in self.weights:
                    expected[f"{prefix}.weight"] = (shape[0], shape[1] // 4)
                    expected[f"{prefix}.scales"] = (shape[0], shape[1] // 64)
                    expected[f"{prefix}.q_biases"] = (shape[0], shape[1] // 64)
                else:
                    expected[f"{prefix}.weight"] = shape
        if self.weights.keys() != expected.keys():
            missing = sorted(expected.keys() - self.weights.keys())
            extra = sorted(self.weights.keys() - expected.keys())
            raise ValueError(
                f"Checkpoint keys do not match GPT-2: missing={missing}, extra={extra}"
            )
        for name, shape in expected.items():
            if self.weights[name].shape != shape:
                raise ValueError(
                    f"Invalid shape for {name}: {self.weights[name].shape}, need {shape}"
                )

    @property
    def weight_bytes(self) -> int:
        return sum(w.nbytes for w in self.weights.values())

    def make_cache(self, capacity: int) -> list[KVCache]:
        if not 1 <= capacity <= self.config.n_positions:
            raise ValueError("Cache capacity must fit the model context window.")
        return [KVCache(capacity) for _ in range(self.config.n_layer)]

    def _linear(self, x: mx.array, name: str) -> mx.array:
        w = self.weights
        if f"{name}.scales" in w:
            y = mx.quantized_matmul(
                x,
                w[f"{name}.weight"],
                w[f"{name}.scales"],
                w[f"{name}.q_biases"],
                transpose=True,
                group_size=64,
                bits=8,
            )
            return y + w[f"{name}.bias"]
        return mx.addmm(w[f"{name}.bias"], x, w[f"{name}.weight"].T)

    def _norm(self, x: mx.array, name: str) -> mx.array:
        return mx.fast.layer_norm(
            x,
            self.weights[f"{name}.weight"],
            self.weights[f"{name}.bias"],
            eps=self.config.layer_norm_epsilon,
        )

    def __call__(
        self, tokens: mx.array, cache: list[KVCache] | None = None, *, last_only: bool = True
    ) -> mx.array:
        c = self.config
        if tokens.ndim != 2 or tokens.shape[0] != 1 or tokens.shape[1] == 0:
            raise ValueError("Expected one non-empty token sequence shaped [1, sequence].")
        if cache is not None and len(cache) != c.n_layer:
            raise ValueError("Expected one KV cache per transformer layer.")
        offset = cache[0].offset if cache else 0
        length = tokens.shape[1]
        if offset + length > c.n_positions:
            raise ValueError(f"GPT-2 context exceeds {c.n_positions} tokens.")
        x = (
            self.weights["wte.weight"][tokens]
            + self.weights["wpe.weight"][mx.arange(offset, offset + length)]
        )
        head_dim = c.n_embd // c.n_head
        for i in range(c.n_layer):
            prefix = f"h.{i}"
            qkv = self._linear(self._norm(x, f"{prefix}.ln_1"), f"{prefix}.attn.c_attn")
            q, k, v = (
                part.reshape(1, length, c.n_head, head_dim).transpose(0, 2, 1, 3)
                for part in mx.split(qkv, 3, axis=-1)
            )
            if cache is not None:
                k, v = cache[i].update(k, v)
            attention = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=head_dim**-0.5, mask="causal" if length > 1 else None
            )
            attention = attention.transpose(0, 2, 1, 3).reshape(1, length, c.n_embd)
            x = x + self._linear(attention, f"{prefix}.attn.c_proj")
            h = self._linear(self._norm(x, f"{prefix}.ln_2"), f"{prefix}.mlp.c_fc")
            h = gelu(h)
            x = x + self._linear(h, f"{prefix}.mlp.c_proj")
        if last_only:
            x = x[:, -1, :]
        x = self._norm(x, "ln_f")
        return x @ self.weights["wte.weight"].T


def load_checkpoint(
    directory: Path,
    precision: str,
    progress: Callable[[str], None] | None = None,
    stop: threading.Event | None = None,
) -> GPT2:
    """Convert one tensor at a time, avoiding a second complete FP32 model in RAM."""
    if precision not in ("fp16", "int8", "fp32"):
        raise ValueError("Precision must be fp16 or int8 (fp32 is for validation).")
    config = ModelConfig.from_file(directory / "config.json")
    dtype = mx.float32 if precision == "fp32" else mx.float16
    weights: dict[str, mx.array] = {}
    with safe_open(directory / "model.safetensors", framework="numpy") as source:
        keys = list(source.keys())
        for i, source_name in enumerate(keys):
            if stop is not None and stop.is_set():
                raise InterruptedError("Model loading cancelled.")
            name = source_name.removeprefix("transformer.")
            if name == "lm_head.weight":
                continue
            # Older checkpoints serialize causal masks; fused attention builds its own mask.
            if name.endswith((".attn.bias", ".attn.masked_bias")):
                continue
            weight = mx.array(source.get_tensor(source_name)).astype(dtype)
            is_linear = name.endswith(
                (
                    ".attn.c_attn.weight",
                    ".attn.c_proj.weight",
                    ".mlp.c_fc.weight",
                    ".mlp.c_proj.weight",
                )
            )
            if is_linear:
                weight = mx.contiguous(weight.T)
            if is_linear and precision == "int8":
                packed, scales, biases = mx.quantize(weight, group_size=64, bits=8)
                mx.eval(packed, scales, biases)
                prefix = name.removesuffix(".weight")
                weights[name] = packed
                weights[f"{prefix}.scales"] = scales
                weights[f"{prefix}.q_biases"] = biases
            else:
                mx.eval(weight)
                weights[name] = weight
            if progress and (i % 24 == 0 or i == len(keys) - 1):
                progress(f"Loading {precision.upper()} weights · {i + 1}/{len(keys)}")
    return GPT2(config, weights)
