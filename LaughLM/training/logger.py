import time
import math
import sys
import jax
from LaughLM.config.schema import LaughLMConfig


# ------------------------------------------------------------
# ANSI helpers
# ------------------------------------------------------------
def _tty():
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

_C = _tty()

def _ansi(code, t):
    return f"\033[{code}m{t}\033[0m" if _C else t

def dim(t):    return _ansi("2", t)
def grey(t):   return _ansi("90", t)
def white(t):  return _ansi("1;37", t)
def green(t):  return _ansi("32", t)
def cyan(t):   return _ansi("36", t)


# ------------------------------------------------------------
# Formatting
# ------------------------------------------------------------
def fmt_time(sec):
    if sec is None or not math.isfinite(sec):
        return "--"
    sec = int(max(sec, 0))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h: return f"{h}h{m:02d}m"
    if m: return f"{m}m{s:02d}s"
    return f"{s}s"


def fmt_tokens(n):
    if n >= 1e9: return f"{n/1e9:.1f}B"
    if n >= 1e6: return f"{n/1e6:.1f}M"
    if n >= 1e3: return f"{n/1e3:.1f}K"
    return str(int(n))


def fmt_ppl(loss):
    try:
        p = math.exp(min(loss, 20))
        if p >= 1000: return f"{p:.0f}"
        return f"{p:.2f}"
    except:
        return "--"


# ------------------------------------------------------------
# Column widths
# ------------------------------------------------------------
_W = dict(
    step=6,
    prog=8,
    loss=9,
    ppl=7,
    gnorm=7,
    lr=12,
    toks=9,
    seen=8,
    rem=9,
    eta=10,
    elapsed=9,
)

SEP = "  " + dim("│") + "  "


def _header():
    return (
        " "
        + f"{'STEP':>{_W['step']}}  {'PROGRESS':>{_W['prog']}}"
        f"  │  "
        f"{'LOSS':>{_W['loss']}}  {'PPL':>{_W['ppl']}}  {'GNORM':>{_W['gnorm']}}"
        f"  │  "
        f"{'LR':>{_W['lr']}}"
        f"  │  "
        f"{'TOK/S':>{_W['toks']}}"
        f"  │  "
        f"{'SEEN':>{_W['seen']}}  {'REMAINING':>{_W['rem']}}  {'ETA':>{_W['eta']}}  {'ELAPSED':>{_W['elapsed']}}"
    )


# ------------------------------------------------------------
# LOGGER
# ------------------------------------------------------------
class TrainingLogger:

    def __init__(self, config: LaughLMConfig, total_params: int):
        self.config = config
        self.total_params = total_params

        from LaughLM.training.scheduler import compute_total_steps
        self.total_steps = compute_total_steps(config)

        self.tokens_per_step = (
            config.runtime.seq_len
            * config.runtime.micro_batch_per_device
            * config.parallelism.data_parallel
            * config.runtime.gradient_accumulation
        )

        self.total_tokens = self.tokens_per_step * self.total_steps

        print(f"[Logger] Tokens/step: {self.tokens_per_step:,}")

        self.start_time = time.time()
        self.last_time = time.time()
        self.last_step = 0

        self.step_time_window = []
        self.best_loss = float("inf")

        self._printed_header = False


    def log_step(self, step, metrics, lr=None, grad_norm=None, tokens_seen=None):

        if step % self.config.runtime.log_interval != 0 or step < 10:
            return

        # -------------------------
        # FORCE TPU SYNC (CRITICAL)
        # -------------------------
        # jax.tree_util.tree_map(
        #     lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
        #     metrics
        # )

        # loss = float(jax.device_get(metrics["loss"]))
        
        metrics = jax.device_get(metrics)
        loss = float(metrics["loss"])

        now = time.time()
        dt = now - self.last_time
        ds = step - self.last_step

        self.last_time = now
        self.last_step = step

        # -------------------------
        # Stable step time
        # -------------------------
        step_time = dt / max(ds, 1)
        self.step_time_window.append(step_time)

        if len(self.step_time_window) > 5:
            self.step_time_window.pop(0)

        avg_step_time = sum(self.step_time_window) / len(self.step_time_window)

        # prevent crazy spikes
        avg_step_time = max(avg_step_time, 1e-4)

        toks_per_sec = self.tokens_per_step / avg_step_time

        # -------------------------
        # Tokens tracking
        # -------------------------
        if tokens_seen is None:
            tokens_seen = step * self.tokens_per_step

        remaining = max(0, self.total_tokens - tokens_seen)
        progress = 100 * step / self.total_steps

        eta = remaining / toks_per_sec if toks_per_sec > 0 else float("nan")
        elapsed = now - self.start_time

        # -------------------------
        # Best marker
        # -------------------------
        is_best = loss < self.best_loss
        if is_best:
            self.best_loss = loss

        marker = green("*") if is_best else " "

        # -------------------------
        # Formatting
        # -------------------------
        c_step = dim(str(step).rjust(_W['step']))
        c_prog = grey(f"{progress:.1f}%".rjust(_W['prog']))

        c_loss = white(f"{loss:.4f}".rjust(_W['loss']))
        c_ppl  = dim(fmt_ppl(loss).rjust(_W['ppl']))
        c_gnorm = grey("n/a".rjust(_W['gnorm']))

        lr_val = lr if lr is not None else 0.0
        c_lr = dim(f"{lr_val:.2e}".rjust(_W['lr']))

        c_toks = dim(f"{int(toks_per_sec):,}".rjust(_W['toks']))

        c_seen = dim(fmt_tokens(tokens_seen).rjust(_W['seen']))
        c_rem  = dim(fmt_tokens(remaining).rjust(_W['rem']))
        c_eta  = grey(fmt_time(eta).rjust(_W['eta']))
        c_elapsed = cyan(fmt_time(elapsed).rjust(_W['elapsed']))

        row = (
            marker
            + c_step + "  " + c_prog
            + SEP
            + c_loss + "  " + c_ppl + "  " + c_gnorm
            + SEP
            + c_lr
            + SEP
            + c_toks
            + SEP
            + c_seen + "  " + c_rem + "  " + c_eta + "  " + c_elapsed
        )

        # -------------------------
        # Header print
        # -------------------------
        if not self._printed_header:
            h = _header()
            print(grey(h))
            print(dim("─" * len(h)))
            self._printed_header = True

        print(row)


    def log_summary(self, step, tokens):
        elapsed = time.time() - self.start_time

        print("\n" + "=" * 60)
        print("Training Summary")
        print("=" * 60)

        print(f"Final step:        {step:,}")
        print(f"Tokens processed:  {tokens:,}")
        print(f"Wall time:         {elapsed/3600:.2f} hours")

        print("=" * 60) 