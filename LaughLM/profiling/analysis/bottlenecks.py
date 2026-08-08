"""
LaughLM/profiling/analysis/bottlenecks.py

Bottleneck analyzer for identifying training pipeline performance bottlenecks.
"""

from __future__ import annotations

from typing import Dict, Any, List
from LaughLM.profiling.core.session import ProfileSession


class BottleneckAnalyzer:
    """
    Analyzes aggregated performance timing metrics to diagnose primary bottlenecks
    and generate actionable optimization recommendations.
    """

    def analyze(
        self,
        aggregated: Dict[str, Any],
        session: ProfileSession,
    ) -> Dict[str, Any]:
        """
        Diagnose performance bottlenecks from aggregated session metrics.
        """
        by_category = aggregated.get("by_category", {})
        by_name = aggregated.get("by_name", {})
        mean_step_ms = aggregated.get("mean_step_ms", 0.0)
        total_steps = aggregated.get("total_steps", 0)

        findings: List[Dict[str, Any]] = []

        # 1. Pipeline-bound (Data loading / prefetch wait)
        data_stats = by_category.get("data") or by_name.get("data_wait") or by_name.get("dataset/prefetch")
        if data_stats and mean_step_ms > 0:
            data_pct = data_stats.get("pct_of_step", (data_stats.get("mean_ms", 0.0) / mean_step_ms) * 100.0)
            if data_pct >= 20.0:
                confidence = "high" if data_pct >= 30.0 else "medium"
                findings.append({
                    "bottleneck": "pipeline-bound",
                    "category": "input pipeline",
                    "confidence": confidence,
                    "evidence": f"Data loading/prefetch wait consumed {data_pct:.1f}% of measured step time ({data_stats.get('mean_ms', 0.0):.2f} ms / {mean_step_ms:.2f} ms).",
                    "pct_of_step": data_pct,
                    "mean_ms": data_stats.get("mean_ms", 0.0),
                })
            elif data_pct >= 10.0:
                findings.append({
                    "bottleneck": "pipeline-bound",
                    "category": "input pipeline",
                    "confidence": "low",
                    "evidence": f"Data loading consumed {data_pct:.1f}% of step time.",
                    "pct_of_step": data_pct,
                    "mean_ms": data_stats.get("mean_ms", 0.0),
                })

        # 2. Host-bound (Host batch preparation, NumPy operations)
        host_stats = by_category.get("host_prepare") or by_name.get("host_prepare")
        if host_stats and mean_step_ms > 0:
            host_pct = host_stats.get("pct_of_step", (host_stats.get("mean_ms", 0.0) / mean_step_ms) * 100.0)
            if host_pct >= 15.0:
                confidence = "high" if host_pct >= 25.0 else "medium"
                findings.append({
                    "bottleneck": "host-bound",
                    "category": "host preparation",
                    "confidence": confidence,
                    "evidence": f"Host batch preparation consumed {host_pct:.1f}% of measured step time ({host_stats.get('mean_ms', 0.0):.2f} ms).",
                    "pct_of_step": host_pct,
                    "mean_ms": host_stats.get("mean_ms", 0.0),
                })

        # 3. Communication-bound (SPMD/FSDP cross-device communication)
        comm_stats = by_category.get("communication") or by_name.get("communication")
        if comm_stats and mean_step_ms > 0:
            comm_pct = comm_stats.get("pct_of_step", (comm_stats.get("mean_ms", 0.0) / mean_step_ms) * 100.0)
            if comm_pct >= 15.0:
                confidence = "high" if comm_pct >= 25.0 else "medium"
                findings.append({
                    "bottleneck": "communication-bound",
                    "category": "interconnect/communication",
                    "confidence": confidence,
                    "evidence": f"Communication overhead consumed {comm_pct:.1f}% of measured step time ({comm_stats.get('mean_ms', 0.0):.2f} ms).",
                    "pct_of_step": comm_pct,
                    "mean_ms": comm_stats.get("mean_ms", 0.0),
                })

        # 4. Compile-bound (JAX JIT tracing and compilation delay)
        step_metrics = aggregated.get("step_metrics", [])
        if len(step_metrics) >= 2:
            step_0_time = step_metrics[0].get("duration_ms", 0.0)
            subsequent_times = [s.get("duration_ms", 0.0) for s in step_metrics[1:]]
            if subsequent_times:
                avg_subsequent = sum(subsequent_times) / len(subsequent_times)
                if avg_subsequent > 0 and (step_0_time / avg_subsequent) >= 4.0:
                    findings.append({
                        "bottleneck": "compile-bound",
                        "category": "JAX compilation",
                        "confidence": "high",
                        "evidence": f"Initial step execution ({step_0_time:.2f} ms) was {(step_0_time / avg_subsequent):.1f}x slower than subsequent steps ({avg_subsequent:.2f} ms), indicating significant JIT compilation delay.",
                        "compilation_spike_ms": step_0_time - avg_subsequent,
                    })

        # 5. Compute-bound (Device step / GPU / TPU execution)
        compute_stats = by_category.get("compute") or by_name.get("device_step")
        if compute_stats and mean_step_ms > 0:
            compute_pct = compute_stats.get("pct_of_step", (compute_stats.get("mean_ms", 0.0) / mean_step_ms) * 100.0)
            if compute_pct >= 70.0 and not any(f["confidence"] == "high" for f in findings if f["bottleneck"] != "compute-bound"):
                findings.append({
                    "bottleneck": "compute-bound",
                    "category": "device compute",
                    "confidence": "high" if compute_pct >= 80.0 else "medium",
                    "evidence": f"Device step execution consumed {compute_pct:.1f}% of step time ({compute_stats.get('mean_ms', 0.0):.2f} ms), indicating compute saturation.",
                    "pct_of_step": compute_pct,
                    "mean_ms": compute_stats.get("mean_ms", 0.0),
                })

        # Determine primary bottleneck
        if not findings:
            primary_bottleneck = "unknown"
            primary_confidence = "low"
            primary_evidence = "Insufficient timing data or balanced workload across measured stages."
        elif len([f for f in findings if f["confidence"] in ("high", "medium")]) > 1:
            primary_bottleneck = "mixed"
            primary_confidence = "medium"
            high_evidence = " | ".join(f["evidence"] for f in findings)
            primary_evidence = f"Multiple bottlenecks identified: {high_evidence}"
        else:
            sorted_findings = sorted(findings, key=lambda f: f.get("pct_of_step", 0.0), reverse=True)
            primary = sorted_findings[0]
            primary_bottleneck = primary["bottleneck"]
            primary_confidence = primary["confidence"]
            primary_evidence = primary["evidence"]

        return {
            "primary_bottleneck": primary_bottleneck,
            "confidence": primary_confidence,
            "evidence": primary_evidence,
            "findings": findings,
            "total_steps_analyzed": total_steps,
        }

    def generate_recommendations(
        self,
        diagnostics: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        """
        Generate actionable recommendations based strictly on diagnostic findings.
        """
        recs: List[Dict[str, str]] = []
        findings = diagnostics.get("findings", [])
        primary = diagnostics.get("primary_bottleneck")

        processed_bottlenecks = set()

        for finding in findings:
            b_type = finding["bottleneck"]
            if b_type in processed_bottlenecks:
                continue
            processed_bottlenecks.add(b_type)

            if b_type == "pipeline-bound":
                recs.append({
                    "bottleneck": "pipeline-bound",
                    "recommendation": "Investigate memmap batching, prefetch depth (e.g. increase prefetch_to_device size), and background worker count in dataset iterator before optimizing model kernels.",
                    "rationale": finding["evidence"],
                })

            elif b_type == "host-bound":
                recs.append({
                    "bottleneck": "host-bound",
                    "recommendation": "Optimize host-side NumPy array creation, type casting, stack/swapaxes operations, and avoid synchronous CPU-GPU array copies.",
                    "rationale": finding["evidence"],
                })

            elif b_type == "compile-bound":
                recs.append({
                    "bottleneck": "compile-bound",
                    "recommendation": "Perform dry-run JIT warmup steps before timing, avoid dynamic sequence length shapes during training, and ensure static array shapes in train_step.",
                    "rationale": finding["evidence"],
                })

            elif b_type == "communication-bound":
                recs.append({
                    "bottleneck": "communication-bound",
                    "recommendation": "Review SPMD device mesh placement, check for non-overlapping communication barriers, and evaluate Sequence or Tensor Parallelism axis rules.",
                    "rationale": finding["evidence"],
                })

            elif b_type == "compute-bound":
                recs.append({
                    "bottleneck": "compute-bound",
                    "recommendation": "Workload is GPU/TPU compute bound. Consider tuning rematerialization policy (e.g. dots_saveable), leveraging bfloat16/float16 compute precision, or enabling fused QKV / SwiGLU operations.",
                    "rationale": finding["evidence"],
                })

        if not recs and primary == "unknown":
            recs.append({
                "bottleneck": "unknown",
                "recommendation": "Run training for additional steps with '--level detailed' or enable layer profiling to capture fine-grained breakdown.",
                "rationale": "No single stage dominated step time sufficiently to trigger high-confidence bottleneck diagnosis.",
            })

        return recs
