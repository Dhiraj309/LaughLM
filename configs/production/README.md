# Production configs

Production overlays define real training stages and their checkpoint/data
locations. The 4B Smol stage is based on the validated 135M PMAP architecture
and uses the mixed `laughlm-v1` tokenized corpus.

`laughlm_v1_135m_fresh_4b.yaml` is the standalone clean-restart config. It
does not require an override and is also the source of truth for strict
checkpoint restore, Hugging Face export, parity testing, and native generation.

`laughlm_v3_138m_balanced_4b.yaml` is the selected balanced production
candidate: 768 hidden units, 18 layers, 12 query heads, and 4 KV heads. It
uses the same 1,048,576-token optimizer batch as the v1 configuration, but
with a 4 microbatch x 16 accumulation geometry. It must start in its own
checkpoint directory and must not resume from a v1/v2 checkpoint.
