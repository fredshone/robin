from typing import Optional

import wandb
from robin.runners.run import run_command


def sweep_command(
    config: dict,
    id: Optional[str] = None,
    count: int = None,
    test: bool = True,
    verbose: bool = False,
):
    import os

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    sweep_cfg = config.pop("sweep")
    project = sweep_cfg.get("project")
    if "seed" not in config:
        config["seed"] = sweep_cfg.get("seed", 42)

    sweep_id = id
    if id is None:
        sweep_id = wandb.sweep(sweep_cfg, project=project)

    def sweep_run():
        run_command(config=config, verbose=verbose, test=test)

    wandb.agent(sweep_id, function=sweep_run, count=count, project=project)
