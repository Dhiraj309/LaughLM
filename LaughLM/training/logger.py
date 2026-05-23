"""
LaughLM/training/logger.py

Training logger with:
- frontier-grade MFU estimation
- async JSONL metrics persistence
- every-step metrics logging
- bounded non-blocking queue
- low-overhead host-side logging
- TPU/GPU-safe runtime semantics

PMAP measurement cleanup:
──────────────────────────────────────────────
1. Logs both PaLM-style non-embedding MFU and logits-inclusive MFU estimate.
2. Separates end-to-end tokens/sec from device-only tokens/sec when provided.
3. Persists timing breakdown:
   - data_wait_time
   - host_batch_prepare_time
   - device_step_time
   - total_step_time
4. Avoids double EMA update when log_metrics() and log_step() are both called.
"""

import json
import math
import queue
import sys
import threading
import time

from pathlib import Path
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
    "a10g": 70.0e12,
    "t4": 65e12,
    "v100": 125e12,
    "l4": 121e12,
    "l40s": 362e12,
    "h100": 989e12,
    "h200": 989e12,
}


def estimate_hardware_flops(
    config: LaughLMConfig,
    num_devices: int = None,
) -> float:
    accel = config.hardware.accelerator
    hw_type = config.hardware.type.lower()

    if num_devices is None:
        num_devices = jax.device_count()

    if accel == "tpu":
        for key, flops in _TPU_FLOPS.items():
            if key in hw_type:
                return flops * num_devices

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

        return _GPU_FLOPS[hw_type] * num_devices

    raise ValueError(
        f"Unknown accelerator: '{accel}'"
    )


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
    return (
        hasattr(sys.stdout, "isatty")
        and sys.stdout.isatty()
    )


_C = _tty()


def _ansi(code, t):
    return f"\033[{code}m{t}\033[0m" if _C else t


def dim(t): return _ansi("2", t)
def grey(t): return _ansi("90", t)
def white(t): return _ansi("1;37", t)
def green(t): return _ansi("32", t)
def cyan(t): return _ansi("36", t)


def fmt_tokens(n):
    n = int(n)

    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.3f}B"

    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"

    if n >= 1_000:
        return f"{n / 1_000:.1f}K"

    return str(n)


def fmt_time(sec):
    sec = max(0, int(sec))

    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)

    if h:
        return f"{h}h{m:02d}m"

    if m:
        return f"{m}m{s:02d}s"

    return f"{s}s"


def fmt_lr(lr):
    return f"{lr:.2e}"


def fmt_ppl(loss):
    p = math.exp(
        min(loss, math.log(9_999_999))
    )

    if p >= 10_000:
        return f"{p / 1000:.1f}K"

    if p >= 1_000:
        return f"{p:.0f}"

    if p >= 100:
        return f"{p:.1f}"

    return f"{p:.2f}"


def fmt_mfu(mfu):
    if mfu >= 10:
        return f"{mfu:.1f}%"

    if mfu >= 1:
        return f"{mfu:.2f}%"

    return f"{mfu:.3f}%"


# ─────────────────────────────────────────────────────────────
# Table layout
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
        f"{'SEEN':>{_W['seen']}} {'REMAINING':>{_W['rem']}} "
        f"{'ETA':>{_W['eta']}} {'ELAPSED':>{_W['elapsed']}}"
    )


_HEADER_PLAIN = _header_plain()
_HEADER = grey(_HEADER_PLAIN)
_RULE = dim("─" * len(_HEADER_PLAIN))


# ─────────────────────────────────────────────────────────────
# Training Logger
# ─────────────────────────────────────────────────────────────

class TrainingLogger:
    def __init__(
        self,
        config: LaughLMConfig,
        total_params: int,
        embedding_params: int,
        num_devices: int = None,
    ):
        self.config = config

        self._num_devices = num_devices or jax.device_count()

        self.total_params = int(total_params)
        self.embedding_params = int(embedding_params)

        self._non_emb_params = (
            self.total_params
            - self.embedding_params
        )

        from LaughLM.training.scheduler import compute_total_steps

        self.total_steps = compute_total_steps(
            config,
            num_devices=self._num_devices,
        )

        self._tokens_total = int(config.runtime.total_tokens)

        self._hw_flops = estimate_hardware_flops(
            config,
            num_devices=self._num_devices,
        )

        self._n_layers = int(config.model.num_layers)
        self._d_model = int(config.model.d_model)
        self._seq_len = int(config.runtime.seq_len)
        self._vocab_size = int(config.model.vocab_size)

        self._micro_batch_per_device = int(
            config.runtime.micro_batch_per_device
        )

        self._grad_accum = int(
            config.runtime.gradient_accumulation
        )

        self._global_batch = (
            self._micro_batch_per_device
            * self._num_devices
        )

        print(
            f"[MFU] Hardware peak: "
            f"{self._hw_flops / 1e12:.1f} TFLOPs bf16 aggregate"
        )

        print(
            f"[MFU] Non-embedding params: "
            f"{self._non_emb_params:,}"
        )

        print(
            f"[MFU] Embedding/logits params: "
            f"{self.embedding_params:,}"
        )

        self.start_time = time.time()

        self._best_loss = float("inf")

        self._ema_toks_per_sec = None
        self._ema_alpha = 0.1

        self._printed_header = False
        self._lines_since_header = 0
        self._header_every = 50

        # ====================================================
        # Async metrics logging
        # ====================================================

        self._is_writer = jax.process_index() == 0

        self._metrics_queue = queue.Queue(
            maxsize=4096
        )

        self._stop_event = threading.Event()

        self._flush_every = 50
        self._pending_writes = 0

        ckpt_dir = Path(config.runtime.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_path = ckpt_dir / "metrics.jsonl"

        self._writer_thread = None

        if self._is_writer:
            self._writer_thread = threading.Thread(
                target=self._writer_loop,
                daemon=True,
            )

            self._writer_thread.start()

    # ========================================================
    # MFU estimates
    # ========================================================

    def _estimate_flops_per_step(
        self,
        tokens_in_step: int,
    ) -> tuple[float, float, float]:
        """
        Returns:
            non_embedding_flops:
                PaLM-style 6 * non_embedding_params * tokens
                plus attention quadratic estimate.

            logits_flops:
                Estimate for tied embedding logits:
                hidden @ embedding.T forward + backward wrt hidden
                + backward wrt embedding ~= 6 * embedding_params * tokens.

            with_logits_flops:
                non_embedding_flops + logits_flops
        """
        tokens = float(tokens_in_step)

        param_flops = (
            6.0
            * float(self._non_emb_params)
            * tokens
        )

        attn_flops = (
            12.0
            * float(self._n_layers)
            * float(self._d_model)
            * float(self._seq_len)
            * tokens
        )

        non_embedding_flops = (
            param_flops
            + attn_flops
        )

        logits_flops = (
            6.0
            * float(self.embedding_params)
            * tokens
        )

        with_logits_flops = (
            non_embedding_flops
            + logits_flops
        )

        return (
            non_embedding_flops,
            logits_flops,
            with_logits_flops,
        )

    # ========================================================
    # Metrics event
    # ========================================================

    def _build_metrics_event(
        self,
        *,
        step: int,
        metrics: Dict,
        lr: float,
        grad_norm: Optional[float],
        tokens_seen: int,
        tokens_in_step: int,
        step_time: float,
        timing_breakdown: Optional[Dict[str, float]] = None,
        update_ema: bool = True,
    ):
        loss = _scalar(
            metrics.get(
                "loss",
                float("nan"),
            )
        )

        timing_breakdown = timing_breakdown or {}

        total_step_time = float(
            timing_breakdown.get(
                "total_step_time",
                step_time,
            )
        )

        data_wait_time = float(
            timing_breakdown.get(
                "data_wait_time",
                0.0,
            )
        )

        host_batch_prepare_time = float(
            timing_breakdown.get(
                "host_batch_prepare_time",
                0.0,
            )
        )

        device_step_time = float(
            timing_breakdown.get(
                "device_step_time",
                step_time,
            )
        )

        total_step_time = max(total_step_time, 1e-6)
        device_step_time = max(device_step_time, 1e-6)

        toks_per_sec = (
            tokens_in_step
            / total_step_time
        )

        device_toks_per_sec = (
            tokens_in_step
            / device_step_time
        )

        if update_ema:
            if self._ema_toks_per_sec is None:
                self._ema_toks_per_sec = toks_per_sec
            else:
                self._ema_toks_per_sec = (
                    self._ema_alpha
                    * toks_per_sec
                    + (1.0 - self._ema_alpha)
                    * self._ema_toks_per_sec
                )

        (
            non_embedding_flops,
            logits_flops,
            with_logits_flops,
        ) = self._estimate_flops_per_step(
            tokens_in_step=tokens_in_step,
        )

        mfu_non_embedding = (
            non_embedding_flops
            / device_step_time
            / self._hw_flops
        ) * 100.0

        mfu_with_logits_estimate = (
            with_logits_flops
            / device_step_time
            / self._hw_flops
        ) * 100.0

        mfu_e2e_non_embedding = (
            non_embedding_flops
            / total_step_time
            / self._hw_flops
        ) * 100.0

        mfu_e2e_with_logits_estimate = (
            with_logits_flops
            / total_step_time
            / self._hw_flops
        ) * 100.0

        ppl = math.exp(
            min(loss, math.log(9_999_999))
        )

        return {
            "step": int(step),
            "loss": float(loss),
            "ppl": float(ppl),
            "grad_norm": (
                None
                if grad_norm is None
                else float(grad_norm)
            ),
            "learning_rate": float(lr),

            # Throughput
            "tokens_per_sec": float(toks_per_sec),
            "device_tokens_per_sec": float(device_toks_per_sec),

            # Backward-compatible alias.
            "mfu": float(mfu_non_embedding),

            # Honest MFU variants.
            "mfu_non_embedding": float(mfu_non_embedding),
            "mfu_with_logits_estimate": float(mfu_with_logits_estimate),
            "mfu_e2e_non_embedding": float(mfu_e2e_non_embedding),
            "mfu_e2e_with_logits_estimate": float(mfu_e2e_with_logits_estimate),

            # FLOP accounting.
            "hardware_peak_tflops": float(self._hw_flops / 1e12),
            "non_embedding_flops_per_step": float(non_embedding_flops),
            "logits_flops_per_step_estimate": float(logits_flops),
            "with_logits_flops_per_step_estimate": float(with_logits_flops),

            # Timing.
            "step_time": float(total_step_time),
            "total_step_time": float(total_step_time),
            "data_wait_time": float(data_wait_time),
            "host_batch_prepare_time": float(host_batch_prepare_time),
            "device_step_time": float(device_step_time),

            # Shape/runtime metadata.
            "tokens_processed": int(tokens_seen),
            "tokens_in_step": int(tokens_in_step),
            "seq_len": int(self._seq_len),
            "global_batch": int(self._global_batch),
            "micro_batch_per_device": int(self._micro_batch_per_device),
            "gradient_accumulation": int(self._grad_accum),
            "num_devices": int(self._num_devices),

            "wall_time": float(time.time()),
        }

    # ========================================================
    # Async queue logging
    # ========================================================

    def log_metrics(
        self,
        *,
        step: int,
        metrics: Dict,
        lr: float,
        grad_norm: Optional[float] = None,
        tokens_seen: int,
        tokens_in_step: int,
        step_time: float,
        timing_breakdown: Optional[Dict[str, float]] = None,
    ):
        if not self._is_writer:
            return

        event = self._build_metrics_event(
            step=step,
            metrics=metrics,
            lr=lr,
            grad_norm=grad_norm,
            tokens_seen=tokens_seen,
            tokens_in_step=tokens_in_step,
            step_time=step_time,
            timing_breakdown=timing_breakdown,
            update_ema=True,
        )

        try:
            self._metrics_queue.put_nowait(event)

        except queue.Full:
            # Never stall training.
            pass

    # ========================================================
    # Console logging
    # ========================================================

    def log_step(
        self,
        *,
        step: int,
        metrics: Dict,
        lr: float,
        grad_norm: Optional[float] = None,
        tokens_seen: Optional[int] = None,
        tokens_in_step: Optional[int] = None,
        step_time: Optional[float] = None,
        timing_breakdown: Optional[Dict[str, float]] = None,
    ):
        if step % self.config.runtime.log_interval != 0:
            return

        if tokens_in_step is None or step_time is None:
            raise ValueError(
                "tokens_in_step and step_time must be provided"
            )

        event = self._build_metrics_event(
            step=step,
            metrics=metrics,
            lr=lr,
            grad_norm=grad_norm,
            tokens_seen=tokens_seen or 0,
            tokens_in_step=tokens_in_step,
            step_time=step_time,
            timing_breakdown=timing_breakdown,
            update_ema=False,
        )

        loss = event["loss"]
        mfu = event["mfu_non_embedding"]
        toks_per_sec = event["tokens_per_sec"]

        remaining = max(
            0,
            self._tokens_total
            - event["tokens_processed"],
        )

        eta = fmt_time(
            remaining
            / max(
                self._ema_toks_per_sec or toks_per_sec,
                1.0,
            )
        )

        elapsed = fmt_time(
            time.time()
            - self.start_time
        )

        pct = (
            100
            * step
            / max(
                self.total_steps,
                1,
            )
        )

        is_best = loss < self._best_loss

        if is_best:
            self._best_loss = loss

        marker = green("*") if is_best else " "

        gnorm_str = (
            f"{grad_norm:.3f}"
            if grad_norm is not None
            else "n/a"
        )

        c_step = dim(str(step).rjust(_W["step"]))
        c_prog = grey(f"{pct:.1f}%".rjust(_W["prog"]))
        c_loss = white(f"{loss:.4f}".rjust(_W["loss"]))
        c_ppl = dim(fmt_ppl(loss).rjust(_W["ppl"]))
        c_gnorm = grey(gnorm_str.rjust(_W["gnorm"]))
        c_lr = dim(fmt_lr(lr).rjust(_W["lr"] - 2))
        c_toks = dim(f"{int(toks_per_sec):,}".rjust(_W["toks"]))
        c_mfu = dim(fmt_mfu(mfu).rjust(_W["mfu"]))

        c_seen = dim(
            fmt_tokens(
                event["tokens_processed"]
            ).rjust(_W["seen"])
        )

        c_rem = dim(
            fmt_tokens(
                remaining
            ).rjust(_W["rem"])
        )

        c_eta = grey(eta.rjust(_W["eta"]))
        c_elap = cyan(elapsed.rjust(_W["elapsed"]))

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

    # ========================================================
    # Writer thread
    # ========================================================

    def _writer_loop(self):
        with open(
            self.metrics_path,
            "a",
            buffering=1,
        ) as f:
            while (
                not self._stop_event.is_set()
                or not self._metrics_queue.empty()
            ):
                try:
                    event = self._metrics_queue.get(
                        timeout=0.1
                    )

                except queue.Empty:
                    continue

                f.write(
                    json.dumps(event)
                    + "\n"
                )

                self._pending_writes += 1

                if self._pending_writes >= self._flush_every:
                    f.flush()
                    self._pending_writes = 0

            f.flush()

    # ========================================================
    # Flush
    # ========================================================

    def flush(self):
        if not self._is_writer:
            return

        while not self._metrics_queue.empty():
            time.sleep(0.01)

    # ========================================================
    # Close
    # ========================================================

    def close(self):
        if not self._is_writer:
            return

        self.flush()

        self._stop_event.set()

        if self._writer_thread is not None:
            self._writer_thread.join()

    # ========================================================
    # Summary
    # ========================================================

    def log_summary(
        self,
        step: int,
        tokens_processed: int,
    ):
        wall_time = max(
            time.time()
            - self.start_time,
            1,
        )

        elapsed = fmt_time(wall_time)

        avg_toks_per_sec = (
            tokens_processed
            / wall_time
        )

        print()
        print(
            dim(
                "="
                * len(_HEADER_PLAIN)
            )
        )

        print(
            white("  Training complete")
            + f"  steps={step:,}"
            + f"  tokens={fmt_tokens(tokens_processed)}"
            + f"  best_loss={self._best_loss:.4f}"
            + f"  avg_tok/s={int(avg_toks_per_sec):,}"
            + f"  elapsed={elapsed}"
        )

        print(
            dim(
                "="
                * len(_HEADER_PLAIN)
            )
        )