import pydantic


class TrainerConfig(pydantic.BaseModel):
    # Helps catch typos / unexpected keys in YAML instead of silently ignoring them

    module: str
    seed: int = 1111
    max_epochs: int = 200
    log_every_n_steps: int = 1
    accelerator: str = "auto"
    precision: str | int | None = None
    checkpoint_interval: int = 50

    # Added to match your YAML
    save_dir: str
    model_name: str
    results_dir: str = "./results"
