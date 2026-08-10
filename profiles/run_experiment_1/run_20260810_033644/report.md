# LaughLM Performance Profile Report

- **Run ID:** `run_20260810_033644`
- **Profile Level:** `developer`
- **Timestamp:** 2026-08-10 03:36:44
- **Steps Profiled:** 86
- **Session Duration:** 301.15 s

## Executive Summary

- **Primary Bottleneck:** `compute-bound` (Confidence: `high`)
- **Evidence:** Device step execution consumed 129.2% of step time (2606.18 ms), indicating compute saturation.
- **Mean Step Time:** `2228.64 ms`
- **Throughput:** `470,500.9 tokens/sec`

## Step Breakdown by Category

| Category | Mean Time (ms) | % of Step | Count | Min (ms) | Max (ms) | p95 (ms) |
|---|---|---|---|---|---|---|
| `step` | 2616.88 | 129.7% | 95 | 2233.78 | 21103.72 | 2236.39 |
| `compute` | 2606.18 | 129.2% | 95 | 2226.91 | 20798.08 | 2228.17 |
| `device_transfer` | 0.82 | 0.0% | 95 | 0.16 | 55.17 | 0.36 |
| `host_prepare` | 0.62 | 0.0% | 95 | 0.40 | 1.73 | 0.96 |
| `data` | 0.26 | 0.0% | 95 | 0.05 | 4.83 | 0.45 |

## Top Invocations by Event Name

| Event Name | Mean Time (ms) | Total Time (ms) | Count | p90 (ms) |
|---|---|---|---|---|
| `step` | 2616.88 | 248603.25 | 95 | 2236.07 |
| `device_step` | 2606.18 | 247587.05 | 95 | 2228.01 |
| `device_put` | 0.82 | 78.36 | 95 | 0.32 |
| `host_prepare` | 0.62 | 59.23 | 95 | 0.84 |
| `data_wait` | 0.26 | 24.37 | 95 | 0.37 |

## Bottleneck Analysis

### Bottleneck: `compute-bound`
- **Category:** device compute
- **Confidence:** `high`
- **Evidence:** Device step execution consumed 129.2% of step time (2606.18 ms), indicating compute saturation.

## Actionable Optimization Recommendations

### 1. `compute-bound`
**Recommendation:** Workload is GPU/TPU compute bound. Consider tuning rematerialization policy (e.g. dots_saveable), leveraging bfloat16/float16 compute precision, or enabling fused QKV / SwiGLU operations.
**Rationale:** Device step execution consumed 129.2% of step time (2606.18 ms), indicating compute saturation.

## Environment & System Info

- **python_version:** `3.12.13 (main, Jun 24 2026, 04:58:23) [GCC 14.2.0]`
- **platform:** `linux`
- **jax_version:** `0.10.2`
- **jax_backend:** `tpu`
- **device_count:** `8`
- **local_device_count:** `8`
- **local_devices:** `['TPU_0(process=0,(0,0,0,0))', 'TPU_1(process=0,(1,0,0,0))', 'TPU_2(process=0,(0,1,0,0))', 'TPU_3(process=0,(1,1,0,0))', 'TPU_4(process=0,(0,2,0,0))', 'TPU_5(process=0,(1,2,0,0))', 'TPU_6(process=0,(0,3,0,0))', 'TPU_7(process=0,(1,3,0,0))']`
