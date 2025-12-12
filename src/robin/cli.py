from typing import Optional

import click
import yaml

from robin.runners.run import run_command
from robin.runners.tune import tune_command


@click.version_option(package_name="mirror")
@click.group()
def cli():
    """Welcome to the mirror cli."""
    pass


@cli.command(name="run")
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--losses", "-l", is_flag=True)
@click.option("--verbose", "-v", is_flag=True)
def run(config_path: click.Path, losses: bool, verbose: bool):
    """Train and report on an encoder and model as per the given configuration file."""

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        run_command(config, verbose=verbose, losses=losses)


@cli.command(name="tune")
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--db-path", "-db", type=click.Path(exists=True))
@click.option("--verbose", "-v", is_flag=True)
def tune(config_path: click.Path, db_path: Optional[click.Path], verbose: bool):
    """Tune a model as per the given configuration file."""

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        tune_command(config, db_path=db_path, verbose=verbose)
