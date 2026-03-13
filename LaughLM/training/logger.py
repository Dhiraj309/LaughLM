"""
LaughLM/training/logger.py

High-performance training logger designed for accelerator workloads.

Goals
-----
• Zero meaningful accelerator overhead
• Clean structured telemetry
• Stable column alignment
• ANSI-safe formatting
• Phase banners for scheduler stages

Critical rule
-------------
Always pad plain strings BEFORE applying ANSI colour.

    WRONG:
        f"{green('text'):>10}"

    RIGHT:
        green(f"{'text':>10}")
"""

import time
import math
import sys
from collections import deque
from typing import Dict, Optional

from LaughLM.config.schema import LaughLMConfig


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
def blue(t):   return _ansi("34", t)


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
# Colour helpers
# ─────────────────────────────────────────────────────────────

def _loss_col(loss):
    if loss > 8.0: return white
    if loss > 5.0: return yellow
    if loss > 3.5: return cyan
    return green

def _gnorm_col(g):
    if g > 3.0: return red
    if g > 1.5: return yellow
    return grey

def _phase_arrow(phase):
    if phase == "warmup": return yellow("↑")
    if phase == "decay":  return cyan("↓")
    return green("—")


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
        f"{'SEEN':>{_W['seen']}}  {'REMAINING':>{_W['rem']}}  {'ETA':>{_W['eta']}}"
    )

_HEADER_PLAIN = _header_plain()
_HEADER       = grey(_HEADER_PLAIN)
_RULE         = dim("─" * len(_HEADER_PLAIN))


# ─────────────────────────────────────────────────────────────
# Instability detector
# ─────────────────────────────────────────────────────────────

class InstabilityDetector:

    def __init__(self, window=50):
        self._loss_buf = deque(maxlen=window)
        self._norm_buf = deque(maxlen=window)
        self._best     = float("inf")

    def check(self, loss, grad_norm):

        if math.isnan(loss):
            return red("⚠ NaN loss detected")

        warnings = []

        if grad_norm and len(self._norm_buf) > 10:
            avg = sum(self._norm_buf) / len(self._norm_buf)
            if grad_norm > 3 * avg:
                warnings.append(
                    yellow(f"grad norm spike: {grad_norm:.3f}")
                )

        self._loss_buf.append(loss)

        if grad_norm:
            self._norm_buf.append(grad_norm)

        return "\n".join(warnings) if warnings else None


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
        self._hw_flops     = 1.576e15

        self._t0        = time.time()
        self._last_t    = time.time()
        self._last_step = 0

        self._window = deque(maxlen=100)

        self._best_loss = float("inf")
        self._phase     = None

        self._detector = InstabilityDetector()

        warmup = config.scheduler.warmup_steps
        stable = int(self.total_steps * getattr(config.scheduler, "stable_fraction", 0.88))

        self._warmup_end = warmup
        self._stable_end = warmup + stable


    # ─────────────────────────────────────────────────────
    # Phase detection
    # ─────────────────────────────────────────────────────

    def _get_phase(self, step):
        if step <= self._warmup_end: return "warmup"
        if step <= self._stable_end: return "stable"
        return "decay"


    def _print_phase_banner(self, phase):

        lr = self.config.optimizer.learning_rate

        if phase == "warmup":
            label = f"steps 1 → {self._warmup_end}"
            lrmsg = f"lr: 0 → {fmt_lr(lr)}"
            colour = yellow

        elif phase == "stable":
            label = f"steps {self._warmup_end+1} → {self._stable_end}"
            lrmsg = f"lr: {fmt_lr(lr)} constant"
            colour = green

        else:
            label = f"steps {self._stable_end+1} → {self.total_steps}"
            lrmsg = f"lr: {fmt_lr(lr)} → 0"
            colour = cyan

        body = f"── {phase.upper()}  [ {label} · {lrmsg} ]"
        banner = body + " " + "─" * max(0, len(_HEADER_PLAIN) - len(body))

        print()
        print(colour(banner))
        print(_HEADER)
        print(_RULE)


    # ─────────────────────────────────────────────────────
    # Main step logger
    # ─────────────────────────────────────────────────────

    def log_step(self, step, metrics, lr, grad_norm=None, tokens_seen=None):

        # 🔴 CRITICAL: only log every N steps
        if step % self.config.runtime.log_interval != 0:
            return

        loss = metrics.get("loss", float("nan"))

        phase = self._get_phase(step)
        if phase != self._phase:
            self._phase = phase
            self._print_phase_banner(phase)

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

        mfu = (6 * self.total_params * avg_toks) / self._hw_flops * 100

        pct = 100 * step / self.total_steps

        is_best = loss < self._best_loss
        if is_best:
            self._best_loss = loss

        marker = green("*") if is_best else " "

        c_step = dim(str(step).rjust(_W['step']))
        c_prog = grey(f"{pct:.1f}%".rjust(_W['prog']))

        c_loss = _loss_col(loss)(f"{loss:.4f}".rjust(_W['loss']))
        c_ppl  = dim(fmt_ppl(loss).rjust(_W['ppl']))

        if grad_norm is None:
            c_gnorm = grey("n/a".rjust(_W['gnorm']))
        else:
            c_gnorm = _gnorm_col(grad_norm)(
                f"{grad_norm:.3f}".rjust(_W['gnorm'])
            )

        lr_val = fmt_lr(lr).rjust(_W['lr']-2)
        c_lr = _phase_arrow(phase) + " " + dim(lr_val)

        c_toks = dim(f"{int(avg_toks):,}".rjust(_W['toks']))
        c_mfu  = dim(fmt_mfu(mfu).rjust(_W['mfu']))

        c_seen = dim(fmt_tokens(tokens_seen).rjust(_W['seen']))
        c_rem  = dim(fmt_tokens(remaining).rjust(_W['rem']))
        c_eta  = grey(eta.rjust(_W['eta']))

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
            + c_seen + "  " + c_rem + "  " + c_eta
        )

        print(row)

        warning = self._detector.check(loss, grad_norm)
        if warning:
            print(f"         {warning}")


    # ─────────────────────────────────────────────────────
    # Training summary
    # ─────────────────────────────────────────────────────

    def log_summary(self, step, tokens_seen):

        elapsed = time.time() - self._t0

        avg_toks = sum(self._window)/len(self._window) if self._window else 0

        w = len(_HEADER_PLAIN)+1

        print()
        print(green("═"*w))
        print(green("Training complete"))
        print(green(f"Steps        {step:,}/{self.total_steps:,}"))
        print(green(f"Tokens seen  {fmt_tokens(tokens_seen)}"))
        print(green(f"Best loss    {self._best_loss:.4f}"))
        print(green(f"Wall time    {fmt_time(elapsed)}"))
        print(green(f"Avg tok/s    {avg_toks:,.0f}"))
        print(green("═"*w))
        print()
