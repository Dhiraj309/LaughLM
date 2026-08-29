# Production configuration

`laughlm_v1_127m_4b.yaml` is the single standalone production configuration
for LaughLM-v1. It trains the 127M GQA model on the mixed `laughlm-v1` corpus.

It is the first cumulative 4B-token milestone of one fixed 20B WSD schedule.
For later milestones, resume from its checkpoint and increase only
`runtime.total_tokens` to `8_000_000_000`, `12_000_000_000`,
`16_000_000_000`, and finally `20_000_000_000`. Keep every other model,
optimizer, scheduler, data, and checkpoint-path setting unchanged.
