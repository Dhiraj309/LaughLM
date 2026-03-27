"""
LaughLM/training/logger.py
"""

import time
import math
import sys
from collections import deque
from typing import Dict

import jax

from LaughLM.config.schema import LaughLMConfig


# ─────────────────────────────────────────────────────────────
# Hardware Peak FLOPs (THEORETICAL, 100%)
# ─────────────────────────────────────────────────────────────

def estimate_hardware_flops(config: LaughLMConfig) -> float:
    accel = config.hardware.accelerator
    hw_type = config.hardware.type.lower()
    devices = config.parallelism.data_parallel

    if accel == "tpu":
        if "v5e" in hw_type:
            per_chip = 197e12
        elif "v4" in hw_type:
            per_chip = 275e12
        else:
            raise ValueError(f"Unknown TPU type: {hw_type}")
        return per_chip * devices

    if accel == "gpu":
        GPU_FLOPS = {
            "t4": 65e12,
            "a100": 312e12,
            "h100": 989e12,
        }

        if hw_type not in GPU_FLOPS:
            raise ValueError(f"Unknown GPU type: {hw_type}")

        return GPU_FLOPS[hw_type] * devices

    raise ValueError(f"Unknown accelerator: {accel}")


# ─────────────────────────────────────────────────────────────
# Scalar safety
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


# ─────────────────────────────────────────────────────────────
# ANSI helpers
# ─────────────────────────────────────────────────────────────

def _tty():
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

_C = _tty()

def _ansi(code, t):
    return f"\033[{code}m{t}\033[0m" if _C else t

def dim(t):    return _ansi("2", t)
def grey(t):   return _ansi("90", t)
def white(t):  return _ansi("1;37", t)
def green(t):  return _ansi("32", t)
def yellow(t): return _ansi("33", t)
def cyan(t):   return _ansi("36", t)
def red(t):    return _ansi("31", t)


# ─────────────────────────────────────────────────────────────
# Formatting utilities
# ─────────────────────────────────────────────────────────────

def fmt_tokens(n):
    if n >= 1_000_000_000: return f"{n/1_000_000_000:.3f}B"
    if n >= 1_000_000:     return f"{n/1_000_000:.1f}M"
    if n >= 1_000:         return f"{n/1_000:.1f}K"
    return str(int(n))

def fmt_time(sec):
    sec = max(0, int(sec))
    h, rem = divmod(sec, 3600)
    m, s   = divmod(rem, 60)
    if h:  return f"{h}h{m:02d}m"
    if m:  return f"{m}m{s:02d}s"
    return f"{s}s"

def fmt_lr(lr):
    return f"{lr:.2e}"

def fmt_ppl(loss):
    p = math.exp(min(loss, math.log(9_999_999)))
    if p >= 100_000: return f"{p/1000:.1f}K"
    if p >= 10_000:  return f"{p/1000:.1f}K"
    if p >= 1_000:   return f"{p:.0f}"
    if p >= 100:     return f"{p:.1f}"
    return f"{p:.2f}"

def fmt_mfu(mfu):
    if mfu >= 10: return f"{mfu:.1f}%"
    if mfu >= 1:  return f"{mfu:.2f}%"
    return        f"{mfu:.3f}%"


# ─────────────────────────────────────────────────────────────
# Column widths
# ─────────────────────────────────────────────────────────────

_W = dict(
    step=6,
    prog=8,
    loss=9,
    ppl=7,
    gnorm=7,
    lr=12,
    toks=7,
    mfu=7,
    seen=8,
    rem=9,
    eta=10,
    elapsed=9,
)

SEP = "  " + dim("│") + "  "


def _header_plain():
    return (
        " "
        + f"{'STEP':>{_W['step']}}  {'PROGRESS':>{_W['prog']}}"
        f"  │  "
        f"{'LOSS':>{_W['loss']}}  {'PPL':>{_W['ppl']}}  {'GNORM':>{_W['gnorm']}}"
        f"  │  "
        f"{'LR':>{_W['lr']}}"
        f"  │  "
        f"{'TOK/S':>{_W['toks']}}  {'MFU':>{_W['mfu']}}"
        f"  │  "
        f"{'SEEN':>{_W['seen']}}  {'REMAINING':>{_W['rem']}}  {'ETA':>{_W['eta']}}  {'ELAPSED':>{_W['elapsed']}}"
    )

_HEADER_PLAIN = _header_plain()
_HEADER       = grey(_HEADER_PLAIN)
_RULE         = dim("─" * len(_HEADER_PLAIN))


# ─────────────────────────────────────────────────────────────
# Main Logger
# ─────────────────────────────────────────────────────────────

class TrainingLogger:

    def __init__(self, config: LaughLMConfig, total_params: int):

        self.config       = config
        self.total_params = total_params

        from LaughLM.training.scheduler import compute_total_steps
        self.total_steps = compute_total_steps(config)

        self._tps = (
            config.runtime.seq_len
            * config.runtime.micro_batch_per_device
            * config.parallelism.data_parallel
            * config.runtime.gradient_accumulation
        )

        self._tokens_total = self.total_steps * self._tps

        # Theoretical peak FLOPs
        self._hw_flops = estimate_hardware_flops(config)
        print(f"[MFU] Theoretical peak FLOPs: {self._hw_flops / 1e12:.2f} TFLOPs")

        self.start_time = time.time()

        self._last_t    = time.time()
        self._last_step = 0
        self._window = deque(maxlen=50)

        self._best_loss = float("inf")

        self._printed_header = False
        self._lines_since_header = 0
        self._header_every = 50


    def log_step(self, step, metrics, lr, grad_norm=None, tokens_seen=None):

        if step % self.config.runtime.log_interval != 0:
            return

        if step < 10:
            return

        loss = _scalar(metrics.get("loss", float("nan")))
        grad_norm = _scalar(grad_norm)

        if tokens_seen is None:
            tokens_seen = step * self._tps

        remaining = max(0, self._tokens_total - tokens_seen)

        now = time.time()
        dt = now - self._last_t
        dsteps = step - self._last_step

        if dt > 0 and dsteps > 0:
            self._window.append((dsteps * self._tps) / dt)

        self._last_t = now
        self._last_step = step

        avg_toks = sum(self._window)/len(self._window) if self._window else 1

        eta = fmt_time(remaining / avg_toks)
        elapsed = fmt_time(now - self.start_time)

        # ------------------------------------------------------------
        # REAL MFU (based on step time)
        # ------------------------------------------------------------
        step_time = dt / max(dsteps, 1)
        flops_per_step = 6 * self.total_params * self._tps

        if step_time > 0:
            real_flops_per_sec = flops_per_step / step_time
            mfu = (real_flops_per_sec / self._hw_flops) * 100
        else:
            mfu = 0.0

        mfu = max(0.0, min(mfu, 100.0))

        pct = 100 * step / self.total_steps

        is_best = loss < self._best_loss
        if is_best:
            self._best_loss = loss

        marker = green("*") if is_best else " "

        c_step = dim(str(step).rjust(_W['step']))
        c_prog = grey(f"{pct:.1f}%".rjust(_W['prog']))

        c_loss = white(f"{loss:.4f}".rjust(_W['loss']))
        c_ppl  = dim(fmt_ppl(loss).rjust(_W['ppl']))

        if grad_norm is None:
            c_gnorm = grey("n/a".rjust(_W['gnorm']))
        else:
            c_gnorm = dim(f"{grad_norm:.3f}".rjust(_W['gnorm']))

        lr_val = fmt_lr(lr).rjust(_W['lr']-2)
        c_lr = dim(lr_val)

        c_toks = dim(f"{int(avg_toks):,}".rjust(_W['toks']))
        c_mfu  = dim(fmt_mfu(mfu).rjust(_W['mfu']))

        c_seen = dim(fmt_tokens(tokens_seen).rjust(_W['seen']))
        c_rem  = dim(fmt_tokens(remaining).rjust(_W['rem']))
        c_eta  = grey(eta.rjust(_W['eta']))
        c_elapsed = cyan(elapsed.rjust(_W['elapsed']))

        row = (
            marker
            + c_step + "  " + c_prog
            + SEP
            + c_loss + "  " + c_ppl + "  " + c_gnorm
            + SEP
            + c_lr
            + SEP
            + c_toks + "  " + c_mfu
            + SEP
            + c_seen + "  " + c_rem + "  " + c_eta + "  " + c_elapsed
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


    def log_summary(self, step: int, tokens: int):

        elapsed = time.time() - self.start_time
        hrs = elapsed / 3600

        print("\n" + "=" * 60)
        print("Training Summary")
        print("=" * 60)

        print(f"Final step:        {step:,}")
        print(f"Tokens processed:  {tokens:,}")
        print(f"Wall time:         {hrs:.2f} hours")

        print("=" * 60)
