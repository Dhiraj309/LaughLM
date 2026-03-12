import time
import jax
import jax.numpy as jnp

from LaughLM.config.schema import LaughLMConfig
from LaughLM.model.gpt import GPTModel
from LaughLM.model.parameter_utils import generate_preflight_report

from LaughLM.training.optimizer import build_optimizer
from LaughLM.training.scheduler import build_scheduler
from LaughLM.training.train_step import create_train_step

from LaughLM.utils.rng import create_rng


class Trainer:
    """
    LaughLM training orchestration.
    """

    def __init__(self, config: LaughLMConfig):
        self.config = LaughLMConfig

        self.rng = create_rng(seed=42)

        generate_preflight_report(config)

        self.model = GPTModel(config=config)

        dummy = jnp.zeros(
            (
                config.runtime.micro_batch_per_device,
                config.runtime.seq_len,
            ),
            dtype=jnp.int32
        )

        params=self.model.init(
            self.rng.next_key(),
            dummy
        )["params"]

        self.params = params

        self.schedule = build_scheduler(config)
        self.optimizer = build_optimizer(config)
        self.opt_state = self.optimizer(params)

        self.train_step = create_train_step(
            self.model, self.optimizer
        )

        self.step = 0

    def train(self, dataloader):
        total_tokens = self.config.runtime.total_tokens

        seq = self.config.runtime.seq_len

        batch = self.config.runtime.micro_batch_per_device

        devices = self.config.parallelism.data_parallel

        tokens_per_step = seq * batch * devices

        total_steps = total_tokens // tokens_per_step


        print(f"\nTraining Steps: {total_steps}\n")

        start = time.time()

        for step, batch_tokens in enumerate(dataloader):
            self.params, self.opt_state, metrics = self.train_step(
                self.params,
                self.opt_state,
                batch_tokens
            )

            self.step += 1

            if self.step % config.runtime.log_interval == 0:
                elapsed = time.time() - start

                tokens_processed = step * tokens_per_step

                tok_per_sec = tokens_processed / max(elapsed, 1)

                lr = self.schedule(step)

                print(
                    f"step={step} "
                    f"loss={metrics['loss']:.4f} "
                    f"lr={lr:.6f} "
                    f"tok/s={tok_per_sec:.0f}"
                )

            if step >= total_steps:
                break
