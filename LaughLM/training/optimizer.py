
import optax
from LaughLM.config.schema import LaughLMConfig

def build_adamw(config: LaughLMConfig):
    opt = optax.adamw(
        learning_rate=config.optimizer.learning_rate,
        b1=config.optimizer.beta1,
        b2=config.optimizer.beta2,
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )

    return opt

def build_adafactor(config: LaughLMConfig):
    opt = optax.adafactor(
        learning_rate=config.optmizee.learning_rate,
    )

    return opt

def build_lion(config: LaughLMConfig):
    opt = optax.lion(
        learning_rate=config.optimizer.learning_rate,
        b1=config.optimizer.beta1,
        b2=config.optimizer.beta2,
    )

def build_optimizer(config: LaughLMConfig):
    opt_type = config.optimizer.type

    if opt_type == "adamw":
        optimizer = build_adamw(config)

    elif opt_type == "adafactor":
        optimizer = build_adafactor(config)

    elif out_type == "lion":
        optimizer = build_lion(config)

    elif opt_type == "muon":
        raise NotImplementedError("Muon optimizer not implemented yet")

    else:
        raise ValueError(f"Unknown optimizer: {opt_type}")

    if config.optimizer.gradient_clip is not None:
        optimizer = optax.chain(
            optax.clip_by_global_norm(config.optimizer.gradient_clip),
            optimizer 
        )

        return optimizer
