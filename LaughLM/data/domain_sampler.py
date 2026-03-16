
import random
import hashlib
from typing import List, Dict, Iterator, Optional, Tuple

from datasets import load_dataset


class DomainSampler:
    """
    Weighted sampler over multiple HuggingFace streaming datasets.

    Features
    --------
    • Supports multiple HF streaming datasets
    • Supports datasets with configs (e.g. wikipedia)
    • Domain-weighted sampling
    • Deterministic shuffling per run
    • Restart-safe streaming
    • Curriculum cooldown phase
    • Optional domain quality scoring
    • Domain statistics tracking
    • Returns (text, domain_id) tuples
    """

    def __init__(
        self,
        sources: List[Dict],
        seed: int = 42,
        shuffle_buffer: int = 10_000,
        min_text_len: int = 200,
        cooldown_sources: Optional[List[Dict]] = None,
        total_steps: Optional[int] = None,
        cooldown_fraction: float = 0.90,
        quality_scorers: Optional[Dict[str, callable]] = None,
    ):

        self.seed = seed
        self.shuffle_buffer = shuffle_buffer
        self.min_text_len = min_text_len

        self.rng = random.Random(seed)

        self._main_sources = sources
        self._cooldown_sources = cooldown_sources or []

        self._total_steps = total_steps
        self._cooldown_at = int((total_steps or 0) * cooldown_fraction)

        self._quality_scorers = quality_scorers or {}

        self.current_step = 0
        self.in_cooldown = False

        self._init_streams(sources)

    # ------------------------------------------------------------
    # Stream initialization
    # ------------------------------------------------------------

    def _init_streams(self, sources: List[Dict]):

        self.domain_names = []
        self.weights = []
        self.datasets = {}

        self._stats: Dict[str, Dict] = {}
        self._source_map = {}

        for src in sources:

            name = src["name"]
            weight = src["weight"]

            self._source_map[name] = src

            ds = self._load_dataset(src)

            self.datasets[name] = iter(ds)

            self.domain_names.append(name)
            self.weights.append(weight)

            self._stats[name] = {
                "yielded": 0,
                "filtered": 0,
                "restarted": 0
            }

        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]

    # ------------------------------------------------------------
    # Stable seed generation
    # ------------------------------------------------------------

    def _stable_hash(self, text: str) -> int:

        h = hashlib.md5(text.encode()).hexdigest()

        return int(h, 16) % 10_000

    # ------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------

    def _load_dataset(self, src: Dict, seed_offset: int = 0):

        name = src["name"]
        config = src.get("config")
        split = src.get("split", "train")

        stable = self._stable_hash(name)

        domain_seed = self.seed + stable + seed_offset

        load_kwargs = dict(split=split, streaming=True)

        if config:
            ds = load_dataset(name, config, **load_kwargs)
        else:
            ds = load_dataset(name, **load_kwargs)

        ds = ds.shuffle(
            buffer_size=self.shuffle_buffer,
            seed=domain_seed
        )

        return ds

    # ------------------------------------------------------------
    # Domain sampling
    # ------------------------------------------------------------

    def _sample_domain(self) -> str:

        return self.rng.choices(
            self.domain_names,
            weights=self.weights
        )[0]

    # ------------------------------------------------------------
    # Restart exhausted dataset
    # ------------------------------------------------------------

    def _restart_dataset(self, name: str):

        src = self._source_map[name]

        ds = self._load_dataset(src, seed_offset=self.current_step)

        self.datasets[name] = iter(ds)

        self._stats[name]["restarted"] += 1

    # ------------------------------------------------------------
    # Text extraction
    # ------------------------------------------------------------

    def _extract_text(self, sample: Dict, domain_id: str) -> str:

        # math datasets
        if "openmath" in domain_id.lower():

            problem = sample.get("problem", "").strip()
            solution = sample.get("solution", "").strip()

            if problem and solution:
                return f"Problem: {problem}\n\nSolution: {solution}"

            return ""

        # conversation datasets
        if "openhermes" in domain_id.lower():

            turns = sample.get("conversations", [])

            parts = []

            for turn in turns:

                role = turn.get("from", "").strip()
                value = turn.get("value", "").strip()

                if role and value:
                    parts.append(f"{role.capitalize()}: {value}")

            return "\n\n".join(parts)

        # code datasets
        if "starcoder" in domain_id.lower() or "the-stack" in domain_id.lower():

            return sample.get("content", "")

        # generic fallback
        for field in ("text", "content", "code", "body"):

            if field in sample and sample[field]:
                return sample[field]

        return ""

    # ------------------------------------------------------------
    # Training step tracking
    # ------------------------------------------------------------

    def step(self):

        self.current_step += 1

        if (
            not self.in_cooldown
            and self._cooldown_sources
            and self._total_steps
            and self.current_step >= self._cooldown_at
        ):

            print(
                f"\n[DomainSampler] Switching to cooldown mix "
                f"at step {self.current_step}/{self._total_steps}"
            )

            self._init_streams(self._cooldown_sources)

            self.in_cooldown = True

    # ------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------

    def print_stats(self):

        print("\n[DomainSampler] Stats")
        print(f"{'Domain':<40} {'Yielded':>10} {'Filtered':>10} {'Restarted':>10}")
        print("-" * 72)

        for name, s in self._stats.items():

            short = name.split("/")[-1][:38]

            print(
                f"{short:<40} "
                f"{s['yielded']:>10,} "
                f"{s['filtered']:>10,} "
                f"{s['restarted']:>10,}"
            )

    # ------------------------------------------------------------
    # Infinite iterator
    # ------------------------------------------------------------

    def __iter__(self) -> Iterator[Tuple[str, str]]:

        while True:

            domain = self._sample_domain()

            domain_id = domain.split("/")[-1]

            try:
                sample = next(self.datasets[domain])

            except StopIteration:

                self._restart_dataset(domain)

                sample = next(self.datasets[domain])

            text = self._extract_text(sample, domain_id)

            if not text or len(text) < self.min_text_len:

                self._stats[domain]["filtered"] += 1

                continue

            scorer = self._quality_scorers.get(domain_id)

            if scorer and not scorer(text):

                self._stats[domain]["filtered"] += 1

                continue

            self._stats[domain]["yielded"] += 1

            yield text, domain_id
