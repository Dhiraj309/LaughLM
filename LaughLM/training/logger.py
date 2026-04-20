"""
LaughLM/training/logger.py
"""

import time
import math
import sys
from typing import Dict, Optional

import jax

from LaughLM.config.schema import LaughLMConfig


# ─────────────────────────────────────────────────────────────
# Hardware Peak FLOPs
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


class TrainingLogger:

    def __init__(self, config: LaughLMConfig, total_params: int, embedding_params: int):

        self.config = config

        self.total_params = total_params
        self.embedding_params = embedding_params

        # ✅ Only transformer params (important fix)
        self._non_emb_params = total_params - embedding_params

        from LaughLM.training.scheduler import compute_total_steps
        self.total_steps = compute_total_steps(config)

        self._tokens_total = config.runtime.total_tokens
        self._hw_flops = estimate_hardware_flops(config)

        print(f"[MFU] Theoretical peak FLOPs: {self._hw_flops / 1e12:.2f} TFLOPs")

        self.start_time = time.time()
        self._best_loss = float("inf")

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

        # 🔥 FIXED FLOPs (correct + realistic)
        flops_per_step = 6 * self._non_emb_params * tokens_in_step

        # 🔥 Add attention FLOPs
        seq_len  = self.config.runtime.seq_len
        n_layers = self.config.model.num_layers
        d_model  = self.config.model.d_model

        attn_flops = 12 * n_layers * d_model * seq_len * tokens_in_step
        flops_per_step += attn_flops

        flops_per_sec = flops_per_step / step_time

        # ✅ Keep clamp (important)
        mfu = max(0.0, min((flops_per_sec / self._hw_flops) * 100, 100.0))

        eta     = fmt_time(remaining / max(toks_per_sec, 1))
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
        elapsed = fmt_time(time.time() - self.start_time)
        toks_per_sec = tokens_processed / max(time.time() - self.start_time, 1)
        print()
        print(dim("=" * len(_HEADER_PLAIN)))
        print(
            white("  Training complete")
            + f"  steps={step:,}"
            + f"  tokens={fmt_tokens(tokens_processed)}"
            + f"  best_loss={self._best_loss:.4f}"
            + f"  tok/s≈{int(toks_per_sec):,}"
            + f"  elapsed={elapsed}"
        )
        print(dim("=" * len(_HEADER_PLAIN)))