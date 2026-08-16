# M7 optional execution matrix

These overlays are experimental and must not replace the native production
configuration. Apply exactly one overlay to
`configs/v5e_pmap_true135m_production.yaml`.

| Track | Overlay | Control | Candidate |
|---|---|---|---|
| Cross-entropy | `v5e_pmap_true135m_native_dense_ce_override.yaml` | native chunked CE | native dense CE |
| Cross-entropy | `v5e_pmap_true135m_tokamax_linear_ce_override.yaml` | native chunked CE | Tokamax linear CE |
| LM-head layout | `v5e_pmap_true135m_untied_lm_head_override.yaml` | tied embedding | untied `[hidden, vocab]` head |
| Fused kernels | `v5e_pmap_true135m_tokamax_kernel_override.yaml` | native kernels | Tokamax dispatcher |
| Layer execution | `v5e_pmap_true135m_scan_layers_override.yaml` | unscanned layers | scanned layers |
| Data loader | `v5e_pmap_true135m_grain_override.yaml` | native memmap | Grain |

The launcher records `loss_contract` in each run manifest. It identifies the
requested backend, Tokamax implementation, fallback policy, and LM-head layout.
Tokamax and Grain remain unaccepted until TPU logs show the requested dispatch,
finite loss, stable resume behavior, and a measured benefit.

## Required comparison controls

- Keep the same model, tokenizer vocabulary, dataset revision, shard selection,
  effective tokens per step, and step window.
- Use separate checkpoint and compilation-cache paths, as provided by each
  overlay.
- Compare steady-state throughput, peak memory, compile time, input wait, loss,
  and checkpoint behavior.
- Treat native fallback as a result to report, not as evidence that the optional
  kernel succeeded.
