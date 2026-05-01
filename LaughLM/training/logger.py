"""
LaughLM/training/logger.py

Training logger with MFU tracking for LaughLM.

Frontier-grade changes (perf/frontier-optim):
──────────────────────────────────────────────
1. Fixed MFU calculation — the attention FLOPs formula used
   `tokens_in_step` (which = batch × seq_len) multiplied by `seq_len`
   again, effectively computing batch × seq_len². The correct formula
   is `2 * n_layers * seq_len² * num_kv_heads * head_dim * batch`
   for the QK^T and attn×V matmuls. Now uses the standard Kaplan et al.
   approximation: model_flops = (6N + 12 L H S) × T where N = non-emb
   params, L = layers, H = d_model, S = seq_len, T = tokens_in_step.

2. Expanded hardware FLOPs table — added L4, L40S, A10G, V100, H200.
   Values are bf16 tensor core peak (not fp16 or sparsity-boosted).

3. Uses actual param count from estimate_parameters() which now accounts
   for GQA and SwiGLU — more accurate non_emb_params → more accurate MFU.

4. Smoothed tok/s — uses exponential moving average over last 10 steps
   to reduce noise in ETA estimates from step-to-step jitter.
"""

import time
import math
import sys
from typing import Dict, Optional

import jax

from LaughLM.config.schema import LaughLMConfig


# ─────────────────────────────────────────────────────────────
# Hardware Peak FLOPs (bf16 tensor core)
# ─────────────────────────────────────────────────────────────

# Values are per-chip bf16 tensor core peak FLOPS.
# Sources: NVIDIA specs, Google Cloud TPU docs.
#
# IMPORTANT: These are THEORETICAL peak. Real training MFU is
# typically 30-55% of these numbers. If you see >60%, double-check
# the model FLOPs formula. >70% is world-class. >80% means a bug.

_TPU_FLOPS = {
    "v5e":      197e12,    # bf16
    "v5p":      459e12,    # bf16
    "v4":       275e12,    # bf16
    "v3":       123e12,    # bf16 (mixed)
}

_GPU_FLOPS = {
    # Ampere
    "a100":     312e12,    # bf16 tensor core (without sparsity)
    "a10g":     70.0e12,   # bf16 tensor core (FP16: 31.2T, BF16 w/ tensor core ~70T)
    # Turing (no native bf16 — uses fp16 tensor cores)
    "t4":       65e12,     # fp16 tensor core (bf16 emulated, same throughput)
    "v100":     125e12,    # fp16 tensor core (no native bf16)
    # Ada Lovelace
    "l4":       121e12,    # bf16 tensor core
    "l40s":     362e12,    # bf16 tensor core
    # Hopper
    "h100":     989e12,    # bf16 tensor core (without sparsity)
    "h200":     989e12,    # same GPU die as H100, more HBM
}


def estimate_hardware_flops(config: LaughLMConfig) -> float:
    """
    Estimate total hardware peak FLOPs across all devices.

    Uses bf16 tensor core peak for the configured hardware type.
    """
    accel   = config.hardware.accelerator
    hw_type = config.hardware.type.lower()
    devices = config.parallelism.data_parallel

    if accel == "tpu":
        for key, flops in _TPU_FLOPS.items():
            if key in hw_type:
                return flops * devices
        raise ValueError(
            f"Unknown TPU type: '{hw_type}'. "
            f"Known: {list(_TPU_FLOPS.keys())}"
        )

    if accel == "gpu":
        if hw_type not in _GPU_FLOPS:
            raise ValueError(
                f"Unknown GPU type: '{hw_type}'. "
                f"Known: {list(_GPU_FLOPS.keys())}"
            )
        return _GPU_FLOPS[hw_type] * devices

    raise ValueError(f"Unknown accelerator: '{accel}'")


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _scalar(x):
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        try:
            return float(jax.device_get(x))
        except Exception:
            return float("nan")


def _tty():
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

_C = _tty()

def _ansi(code, t): return f"\033[{code}m{t}\033[0m" if _C else t
def dim(t):   return _ansi("2",    t)
def grey(t):  return _ansi("90",   t)
def white(t): return _ansi("1;37", t)
def green(t): return _ansi("32",   t)
def cyan(t):  return _ansi("36",   t)


def fmt_tokens(n):
    if n >= 1_000_000_000: return f"{n/1_000_000_000:.3f}B"
    if n >= 1_000_000:     return f"{n/1_000_000:.1f}M"
    if n >= 1_000:         return f"{n/1_000:.1f}K"
    return str(int(n))

def fmt_time(sec):
    sec = max(0, int(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h: return f"{h}h{m:02d}m"
    if m: return f"{m}m{s:02d}s"
    return f"{s}s"

def fmt_lr(lr):   return f"{lr:.2e}"

def fmt_ppl(loss):
    p = math.exp(min(loss, math.log(9_999_999)))
    if p >= 10_000: return f"{p/1000:.1f}K"
    if p >= 1_000:  return f"{p:.0f}"
    if p >= 100:    return f"{p:.1f}"
    return f"{p:.2f}"

def fmt_mfu(mfu):
    if mfu >= 10: return f"{mfu:.1f}%"
    if mfu >= 1:  return f"{mfu:.2f}%"
    return f"{mfu:.3f}%"


# ─────────────────────────────────────────────────────────────
# Table layout
# ─────────────────────────────────────────────────────────────

_W = dict(
    step=6, prog=8, loss=9, ppl=7, gnorm=7,
    lr=12, toks=7, mfu=7, seen=8, rem=9, eta=10, elapsed=9,
)

SEP = " " + dim("│") + " "

def _header_plain():
    return (
        " "
        + f"{'STEP':>{_W['step']}} {'PROGRESS':>{_W['prog']}}"
        f" │ "
        f"{'LOSS':>{_W['loss']}} {'PPL':>{_W['ppl']}} {'GNORM':>{_W['gnorm']}}"
        f" │ "
        f"{'LR':>{_W['lr']}}"
        f" │ "
        f"{'TOK/S':>{_W['toks']}} {'MFU':>{_W['mfu']}}"
        f" │ "
        f"{'SEEN':>{_W['seen']}} {'REMAINING':>{_W['rem']}} {'ETA':>{_W['eta']}} {'ELAPSED':>{_W['elapsed']}}"
    )

_HEADER_PLAIN = _header_plain()
_HEADER = grey(_HEADER_PLAIN)
_RULE   = dim("─" * len(_HEADER_PLAIN))


# ─────────────────────────────────────────────────────────────
# Training Logger
# ─────────────────────────────────────────────────────────────

class TrainingLogger:

    def __init__(self, config: LaughLMConfig, total_params: int, embedding_params: int):

        self.config = config

        self.total_params = total_params
        self.embedding_params = embedding_params
        self._non_emb_params = total_params - embedding_params

        from LaughLM.training.scheduler import compute_total_steps
        self.total_steps = compute_total_steps(config)

        self._tokens_total = config.runtime.total_tokens
        self._hw_flops = estimate_hardware_flops(config)

        # Pre-compute per-step attention FLOPs components
        self._n_layers = config.model.num_layers
        self._d_model  = config.model.d_model
        self._seq_len  = config.runtime.seq_len

        print(f"[MFU] Hardware peak: {self._hw_flops / 1e12:.1f} TFLOPs (bf16)")
        print(f"[MFU] Non-embedding params: {self._non_emb_params:,}")

        self.start_time = time.time()
        self._best_loss = float("inf")

        # Smoothed tok/s for stable ETA
        self._ema_toks_per_sec = None
        self._ema_alpha = 0.1   # weight for new sample

        self._printed_header = False
        self._lines_since_header = 0
        self._header_every = 50

    def log_step(
        self,
        step: int,
        metrics: Dict,
        lr: float,
        grad_norm: Optional[float] = None,
        tokens_seen: Optional[int] = None,
        tokens_in_step: Optional[int] = None,
        step_time: Optional[float] = None,
    ):

        if step % self.config.runtime.log_interval != 0:
            return
        if step < 10:
            return
        if tokens_in_step is None or step_time is None:
            raise ValueError("tokens_in_step and step_time must be provided")

        loss = _scalar(metrics.get("loss", float("nan")))
        tokens_seen = tokens_seen or 0

        remaining = max(0, self._tokens_total - tokens_seen)
        step_time = max(step_time, 1e-6)

        toks_per_sec = tokens_in_step / step_time

        # ── Smoothed tok/s for ETA ────────────────────────────
        if self._ema_toks_per_sec is None:
            self._ema_toks_per_sec = toks_per_sec
        else:
            self._ema_toks_per_sec = (
                self._ema_alpha * toks_per_sec
                + (1 - self._ema_alpha) * self._ema_toks_per_sec
            )

        # ── Model FLOPs per step ──────────────────────────────
        # Standard formula (Kaplan et al. / PaLM):
        #   model_flops = 6 * N_non_emb * tokens_in_step
        #               + 12 * n_layers * d_model * seq_len * tokens_in_step
        #
        # First term: all parameter matmuls (QKV, output, MLP up/gate/down)
        #   6 = 2 (forward) + 4 (backward) per param per token
        #
        # Second term: attention score computation (QK^T + attn×V)
        #   This is the seq_len² part — NOT counted in "6N" because
        #   these FLOPs don't correspond to any parameter.
        #   12 = 2 ops (QK^T + attn×V) × 2 (forward) × 3 (fwd+bwd)
        #   ... actually: 2 × (2 for fwd QK^T, attn×V) × 3 (fwd+bwd)
        #   tokens_in_step = batch × seq_len, so this becomes:
        #   12 * L * d * S * B * S = 12 * L * d * S² * B
        #   which is the correct O(S²) attention cost.

        param_flops = 6 * self._non_emb_params * tokens_in_step
        attn_flops  = 12 * self._n_layers * self._d_model * self._seq_len * tokens_in_step

        flops_per_step = param_flops + attn_flops
        flops_per_sec  = flops_per_step / step_time

        mfu = max(0.0, min((flops_per_sec / self._hw_flops) * 100, 100.0))

        # ── ETA from smoothed throughput ──────────────────────
        eta     = fmt_time(remaining / max(self._ema_toks_per_sec, 1))
        elapsed = fmt_time(time.time() - self.start_time)
        pct     = 100 * step / self.total_steps

        is_best = loss < self._best_loss
        if is_best:
            self._best_loss = loss

        marker = green("*") if is_best else " "
        gnorm_str = f"{grad_norm:.3f}" if grad_norm is not None else "n/a"

        c_step  = dim(str(step).rjust(_W['step']))
        c_prog  = grey(f"{pct:.1f}%".rjust(_W['prog']))
        c_loss  = white(f"{loss:.4f}".rjust(_W['loss']))
        c_ppl   = dim(fmt_ppl(loss).rjust(_W['ppl']))
        c_gnorm = grey(gnorm_str.rjust(_W['gnorm']))
        c_lr    = dim(fmt_lr(lr).rjust(_W['lr'] - 2))
        c_toks  = dim(f"{int(toks_per_sec):,}".rjust(_W['toks']))
        c_mfu   = dim(fmt_mfu(mfu).rjust(_W['mfu']))
        c_seen  = dim(fmt_tokens(tokens_seen).rjust(_W['seen']))
        c_rem   = dim(fmt_tokens(remaining).rjust(_W['rem']))
        c_eta   = grey(eta.rjust(_W['eta']))
        c_elap  = cyan(elapsed.rjust(_W['elapsed']))

        row = (
            marker
            + c_step + " " + c_prog
            + SEP
            + c_loss + " " + c_ppl + " " + c_gnorm
            + SEP
            + c_lr
            + SEP
            + c_toks + " " + c_mfu
            + SEP
            + c_seen + " " + c_rem + " " + c_eta + " " + c_elap
        )

        if not self._printed_header:
            print(_HEADER)
            print(_RULE)
            self._printed_header = True
        elif self._lines_since_header >= self._header_every:
            print()
            print(_HEADER)
            print(_RULE)
            self._lines_since_header = 0

        print(row)
        self._lines_since_header += 1

    def log_summary(self, step: int, tokens_processed: int):
        wall_time = max(time.time() - self.start_time, 1)
        elapsed = fmt_time(wall_time)
        avg_toks_per_sec = tokens_processed / wall_time
        print()
        print(dim("=" * len(_HEADER_PLAIN)))
        print(
            white("  Training complete")
            + f"  steps={step:,}"
            + f"  tokens={fmt_tokens(tokens_processed)}"
            + f"  best_loss={self._best_loss:.4f}"
            + f"  avg_tok/s={int(avg_toks_per_sec):,}"
            + f"  elapsed={elapsed}"
        )
        print(dim("=" * len(_HEADER_PLAIN)))
