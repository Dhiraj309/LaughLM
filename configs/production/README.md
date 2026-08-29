# Production configs

Production overlays define real training stages and their checkpoint/data
locations. The 4B Smol stage is based on the validated 135M PMAP architecture
and uses the mixed `laughlm-v1` tokenized corpus.

`laughlm_v1_135m_fresh_4b.yaml` is the standalone clean-restart config. It
does not require an override and is also the source of truth for strict
checkpoint restore, Hugging Face export, parity testing, and native generation.
