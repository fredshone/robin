from typing import Optional

import click
import yaml

import wandb
from robin.runners.run import print_summary, run_command
from robin.runners.tune import tune_command


@click.version_option(package_name="mirror")
@click.group()
def cli():
    """Welcome to the mirror cli."""
    pass


@cli.command(name="run")
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--test", "-t", is_flag=True)
@click.option("--verbose", "-v", is_flag=True)
@click.option("--offline", "-o", is_flag=True)
def run(config_path: click.Path, test: bool, verbose: bool, offline: bool):
    """Train and report on an encoder and model as per the given configuration file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    logger = run_command(config, verbose=verbose, test=test, offline=offline)
    print_summary(logger)


@cli.command("sweep")
@click.argument("config_path", type=click.Path(exists=True))
@click.option(
    "--count",
    "-c",
    type=int,
    default=None,
    help="Optional: limit number of runs for this agent.",
)
@click.option("--test", "-t", is_flag=True)
@click.option("--verbose", "-v", is_flag=True)
def sweep(config_path: click.Path, count: int, test: bool, verbose: bool):
    """
    Start a W&B sweep using a sweep config.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    sweep_cfg = config.pop("sweep")
    project = sweep_cfg.get("project")

    sweep_id = wandb.sweep(sweep_cfg)

    def sweep_run():
        run_command(config=config, verbose=verbose, test=test)

    wandb.agent(sweep_id, function=sweep_run, count=count, project=project)


@cli.command(name="tune")
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--db-path", "-db", type=click.Path(exists=True))
@click.option("--verbose", "-v", is_flag=True)
def tune(config_path: click.Path, db_path: Optional[click.Path], verbose: bool):
    """Tune a model as per the given configuration file."""

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        tune_command(config, db_path=db_path, verbose=verbose)
