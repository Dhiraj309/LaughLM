# LaughLM dependency lanes

The repository has two dependency concepts:

- `inputs/` contains the small, human-reviewed resolver inputs.
- `locks/` contains the checked-in constraints or generated locks used by CI
  and TPU runbooks.

The current stable CPU lane preserves the versions already declared by
`pyproject.toml`:

```bash
python -m pip install -c requirements/locks/legacy-cpu-py312.txt -e ".[dev]"
python -m pip check
pytest -q
```

`legacy-cpu-py312.txt` is a baseline constraint file, not yet a complete
transitive hash lock. G2-004 will add reproducible hash generation and the
final lock files. The Gen-2 dependency lane is intentionally separate and must
not be installed by the stable command above.
