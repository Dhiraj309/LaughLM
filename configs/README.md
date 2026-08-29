# LaughLM configurations

The production source of truth is:

```text
configs/production/laughlm_v1_127m_4b.yaml
```

It is a standalone configuration—do not combine it with an override config.
The current run stops at 4B cumulative tokens, while its WSD schedule remains
fixed at 20B tokens for safe staged resume.

The remaining root YAML files are test or smoke fixtures retained for automated
coverage and specialized developer scripts; they are not production options.
