import optax

from LaughLM.config.schema import LaughLMConfig

def build_cosine_scheduler(config: LaughLMConfig):

    warmup = config.scheduler.warmup_steps
    lr = config.optimizer.learning_rate
    min_ratio = config.scheduler.min_lr_ratio

    total_steps = (
        config.runtime.total_tokens //
        (config.runtime.seq_len *
         config.runtime.micro_batch_per_device *
         config.parallelism.data_parallel)
    )

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=lr,
        warmup_steps=warmup,
        decay_steps=total_steps,
        end_value=lr * min_ratio
    )

    return schedule

def build_linear_scheduler(config: LaughLMConfig):

    warmup = config.scheduler.warmup_steps
    lr = config.optimizer.learning_rate

    total_steps = (
        config.runtime.total_tokens //
        (config.runtime.seq_len *
         config.runtime.micro_batch_per_device *
         config.parallelism.data_parallel)
    )

    schedule = optax.linear_schedule(
        init_value=0.0,
        end_value=lr,
        transition_steps=warmup
    )

    decay = optax.linear_schedule(
        init_value=lr,
        end_value=0.0,
        transition_steps=total_steps - warmup
    )

    return optax.join_schedules(
        [schedule, decay],
        [warmup]
    )


def build_rsqrt_scheduler(config: LaughLMConfig):

    warmup = config.scheduler.warmup_steps
    lr = config.optimizer.learning_rate

    def schedule(step):

        step = max(step, 1)

        scale = min(
            step ** -0.5,
            step * warmup ** -1.5
        )

        return lr * scale

    return schedule

def build_wsd_scheduler(config: LaughLMConfig):

    warmup = config.scheduler.warmup_steps
    lr = config.optimizer.learning_rate
    min_ratio = config.scheduler.min_lr_ratio

    total_steps = (
        config.runtime.total_tokens //
        (config.runtime.seq_len *
         config.runtime.micro_batch_per_device *
         config.parallelism.data_parallel)
    )

    stable_steps = int(total_steps * 0.6)

    warmup_sched = optax.linear_schedule(
        init_value=0.0,
        end_value=lr,
        transition_steps=warmup
    )

    stable_sched = optax.constant_schedule(lr)

    decay_sched = optax.linear_schedule(
        init_value=lr,
        end_value=lr * min_ratio,
        transition_steps=total_steps - warmup - stable_steps
    )

    return optax.join_schedules(
        [warmup_sched, stable_sched, decay_sched],
        [warmup, warmup + stable_steps]
    )


def build_scheduler(config: LaughLMConfig):

    sched_type = config.scheduler.type

    if sched_type == "cosine":
        return build_cosine_scheduler(config)

    if sched_type == "linear":
        return build_linear_scheduler(config)

    if sched_type == "rsqrt":
        return build_rsqrt_scheduler(config)

    if sched_type == "wsd":
        return build_wsd_scheduler(config)

    raise ValueError(f"Unknown scheduler: {sched_type}")
