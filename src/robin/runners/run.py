from pathlib import Path

import polars as pl
import torch
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torch.random import seed as seeder

from robin.dataloaders.loader import DataModule
from robin.encoders import TableEncoder, YXDataset
from robin.runners import helpers


def run_command(
    config: dict,
    verbose: bool = False,
    ckpt_path: Path = None,
    test: bool = True,
    offline: bool = False,
) -> None:
    """
    Train, test and generate.

    Args:
        config (dict): The configuration dictionary.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
        ckpt_path (Path, optional): Path to a checkpoint file. Defaults to None.
        test (bool, optional): Whether to run tests and return losses. Defaults to True.
        offline (bool, optional): Run logger in offline mode. Defaults to False.

    Returns:
        None

    """
    save_dir = Path(config.get("logging", {}).get("dir", "logs"))
    project = str(config.get("logging", {}).get("project"))

    # create directories
    save_dir.mkdir(exist_ok=True, parents=True)

    # save config
    with open(save_dir / "config.yaml", "w") as file:
        yaml.dump(config, file)

    logger = WandbLogger(
        save_dir=save_dir, project=project, config=config, offline=offline
    )
    config = dict(logger.experiment.config)

    seed = config.pop("seed", seeder())
    torch.manual_seed(seed)

    yx = helpers.load_data(config)
    y, x = helpers.split_data(config, yx)
    x_encoder = TableEncoder(x, verbose=verbose)
    x_dataset = x_encoder.encode(data=x)
    y_encoder = TableEncoder(y, verbose=verbose)
    y_dataset = y_encoder.encode(data=y)

    yx_dataset = YXDataset(y_dataset, x_dataset)
    datamodule = DataModule(dataset=yx_dataset, **config.get("datamodule", {}))

    model = helpers.build_model(
        config=config,
        x_encoder=x_encoder,
        y_encoder=y_encoder,
        ckpt_path=ckpt_path,
        verbose=verbose,
    )

    callbacks = helpers.build_callbacks(config)

    trainer = Trainer(
        callbacks=callbacks, logger=logger, **config.get("trainer", {})
    )

    trainer.fit(model=model, train_dataloaders=datamodule)

    root = Path(trainer.checkpoint_callback.dirpath).parent

    if test:
        loss = helpers.run_tests(trainer=trainer, ckpt_path=ckpt_path)
        logger.experiment.summary.update(loss)

    data_dir = root / "data"
    data_dir.mkdir(exist_ok=True)

    gen_loader = helpers.build_gen_loader(config, y_dataset=y_dataset)

    ys, xs, zs = helpers.generate(
        config, dataloader=gen_loader, trainer=trainer
    )
    xs = helpers.sample(config, xs)

    y_synth = y_encoder.decode(ys)
    x_synth = x_encoder.decode(xs)
    synth = pl.concat([y_synth, x_synth], how="horizontal")

    synth.write_csv(data_dir / "synthetic.csv")

    pl.DataFrame(zs.detach().numpy()).write_csv(
        data_dir / "zs.csv", include_header=False
    )

    metrics = helpers.evaluate(target=yx, synthetic=synth, config=config)
    logger.experiment.summary.update(metrics)

    with open(root / "summary.yaml", "w") as file:
        yaml.dump(logger.experiment.summary, file)

    return logger


def print_summary(logger: WandbLogger):

    print("\n=== Summary ===")
    for k, v in dict(logger.experiment.summary).items():
        if v is None:
            print(f"  {k:>18}: (none)")
        elif isinstance(v, float):
            # Format floats with reasonable precision
            if "accuracy" in k or "meta_score" in k:
                print(f"  {k:>18}: {v:.6f}")
            elif "minutes" in k:
                print(f"  {k:>18}: {v:.2f}")
            else:
                print(f"  {k:>18}: {v:.6f}")
        else:
            print(f"  {k:>18}: {v}")
