"""
LaughLM/training/logger.py

Training logger with MFU tracking for LaughLM.

Key fixes:
──────────
1. tokens_seen is now ALWAYS computed as:
      step × tokens_in_step
   → guarantees global (multi-device correct)

2. Prevents per-device vs global mismatch in logs

3. MFU + throughput logic unchanged (already correct)
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

_TPU_FLOPS = {
    "v5e": 197e12,
    "v5p": 459e12,
    "v4": 275e12,
    "v3": 123e12,
}

_GPU_FLOPS = {
    "a100": 312e12,
    "a10g": 70e12,
    "t4": 65e12,
    "v100": 125e12,
    "l4": 121e12,
    "l40s": 362e12,
    "h100": 989e12,
    "h200": 989e12,
}


def estimate_hardware_flops(config: LaughLMConfig) -> float:
    accel = config.hardware.accelerator
    hw_type = config.hardware.type.lower()
    devices = config.parallelism.data_parallel

    if accel == "tpu":
        for key, flops in _TPU_FLOPS.items():
            if key in hw_type:
                return flops * devices
        raise ValueError(f"Unknown TPU type: {hw_type}")

    if accel == "gpu":
        if hw_type not in _GPU_FLOPS:
            raise ValueError(f"Unknown GPU type: {hw_type}")
        return _GPU_FLOPS[hw_type] * devices

    raise ValueError(f"Unknown accelerator: {accel}")


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
def dim(t): return _ansi("2", t)
def grey(t): return _ansi("90", t)
def white(t): return _ansi("1;37", t)
def green(t): return _ansi("32", t)
def cyan(t): return _ansi("36", t)


def fmt_tokens(n):
    if n >= 1_000_000_000: return f"{n/1e9:.3f}B"
    if n >= 1_000_000: return f"{n/1e6:.1f}M"
    if n >= 1_000: return f"{n/1e3:.1f}K"
    return str(int(n))


def fmt_time(sec):
    sec = max(0, int(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h: return f"{h}h{m:02d}m"
    if m: return f"{m}m{s:02d}s"
    return f"{s}s"


def fmt_lr(lr): return f"{lr:.2e}"


def fmt_ppl(loss):
    p = math.exp(min(loss, math.log(9_999_999)))
    if p >= 10_000: return f"{p/1000:.1f}K"
    if p >= 1_000: return f"{p:.0f}"
    if p >= 100: return f"{p:.1f}"
    return f"{p:.2f}"


def fmt_mfu(mfu):
    if mfu >= 10: return f"{mfu:.1f}%"
    if mfu >= 1: return f"{mfu:.2f}%"
    return f"{mfu:.3f}%"


# ─────────────────────────────────────────────────────────────
# Layout
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
_RULE = dim("─" * len(_HEADER_PLAIN))


# ─────────────────────────────────────────────────────────────
# Logger
# ─────────────────────────────────────────────────────────────

class TrainingLogger:

    def __init__(self, config: LaughLMConfig, total_params: int, embedding_params: int):
        self.config = config
        self._non_emb_params = total_params - embedding_params

        from LaughLM.training.scheduler import compute_total_steps
        self.total_steps = compute_total_steps(config)

        self._tokens_total = config.runtime.total_tokens
        self._hw_flops = estimate_hardware_flops(config)

        self._n_layers = config.model.num_layers
        self._d_model = config.model.d_model
        self._seq_len = config.runtime.seq_len

        print(f"[MFU] Hardware peak: {self._hw_flops / 1e12:.1f} TFLOPs (bf16)")
        print(f"[MFU] Non-embedding params: {self._non_emb_params:,}")

        self.start_time = time.time()
        self._best_loss = float("inf")

        self._ema_toks_per_sec = None
        self._ema_alpha = 0.1

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

        # 🔥 FIX: always use global tokens
        tokens_seen = step * tokens_in_step

        remaining = max(0, self._tokens_total - tokens_seen)
        step_time = max(step_time, 1e-6)

        toks_per_sec = tokens_in_step / step_time

        if self._ema_toks_per_sec is None:
            self._ema_toks_per_sec = toks_per_sec
        else:
            self._ema_toks_per_sec = (
                self._ema_alpha * toks_per_sec
                + (1 - self._ema_alpha) * self._ema_toks_per_sec
            )

        param_flops = 6 * self._non_emb_params * tokens_in_step
        attn_flops = 12 * self._n_layers * self._d_model * self._seq_len * tokens_in_step

        flops_per_step = param_flops + attn_flops
        flops_per_sec = flops_per_step / step_time

        mfu = max(0.0, min((flops_per_sec / self._hw_flops) * 100, 100.0))

        eta = fmt_time(remaining / max(self._ema_toks_per_sec, 1))
        elapsed = fmt_time(time.time() - self.start_time)
        pct = 100 * step / self.total_steps

        if loss < self._best_loss:
            self._best_loss = loss
            marker = green("*")
        else:
            marker = " "

        row = (
            marker
            + dim(str(step).rjust(_W['step'])) + " "
            + grey(f"{pct:.1f}%".rjust(_W['prog']))
            + SEP
            + white(f"{loss:.4f}".rjust(_W['loss'])) + " "
            + dim(fmt_ppl(loss).rjust(_W['ppl'])) + " "
            + grey((f"{grad_norm:.3f}" if grad_norm else "n/a").rjust(_W['gnorm']))
            + SEP
            + dim(fmt_lr(lr).rjust(_W['lr'] - 2))
            + SEP
            + dim(f"{int(toks_per_sec):,}".rjust(_W['toks'])) + " "
            + dim(fmt_mfu(mfu).rjust(_W['mfu']))
            + SEP
            + dim(fmt_tokens(tokens_seen).rjust(_W['seen'])) + " "
            + dim(fmt_tokens(remaining).rjust(_W['rem'])) + " "
            + grey(eta.rjust(_W['eta'])) + " "
            + cyan(elapsed.rjust(_W['elapsed']))
        )

        if not self._printed_header:
            print(_HEADER)
            print(_RULE)
            self._printed_header = True

        print(row)

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
