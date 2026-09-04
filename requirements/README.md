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

The Gen-2 inputs use the modern JAX ecosystem without changing the stable
package metadata. Create a separate Python 3.12 environment, install one of
the inputs, and then install LaughLM without asking its legacy dependency
metadata to resolve again:

```bash
python -m pip install -r requirements/inputs/gen2-cpu-py312.in
python -m pip install --no-deps -e .
```

For a TPU v5e host, use the TPU input instead:

```bash
python -m pip install -r requirements/inputs/gen2-tpu-v5e-py312.in
python -m pip install --no-deps -e .
python -m pip check
```

These inputs are candidate compatibility lanes, not production locks. Do not
mix them with the legacy lane in one environment, and do not call them
production-ready until G2-004 and the migration gates pass.

After installing the pinned lock-tool input, generate the CPU and TPU locks
from the repository root:

```bash
python -m pip install -r requirements/inputs/lock-tools-py312.in
python -m scripts.generate_dependency_locks \
  --input requirements/inputs/gen2-cpu-py312.in \
  --output requirements/locks/gen2-cpu-py312.txt
python -m scripts.generate_dependency_locks \
  --input requirements/inputs/gen2-tpu-v5e-py312.in \
  --output requirements/locks/gen2-tpu-v5e-py312.txt
```

The generator uses pip-tools backtracking resolution, requires a SHA-256 hash
for every resolved package, records its exact version, and writes atomically.
Use `--force` only when deliberately regenerating an existing lock.
