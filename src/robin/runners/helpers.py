from pathlib import Path
from typing import Optional, Tuple

import polars as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch import Tensor, argmax, concat, isnan, multinomial, stack
from torch.utils.data import DataLoader

from robin.encoders import YZDataset, ZDataset
from robin.eval import binning, correctness, creativity, density
from robin.models.cvae import CVAE
from robin.models.cvae_components import (
    ControlsEncoderBlock,
    CVAEDecoderBlock,
    CVAEEncoderBlock,
)
from robin.runners import defaults


def load_data(config: dict) -> pl.DataFrame:
    # load data
    x_path = config["data"]["train_path"]
    x = pl.read_csv(x_path)

    # rename columns and select as required
    rename = config["data"].get("columns")
    if isinstance(rename, dict):
        x = x.rename(rename)
        x = x.select(list(rename.values()))
    elif isinstance(rename, list) and rename:
        x = x.select(rename)
    return x


def split_data(
    config: dict, x: pl.DataFrame
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    y_cols = config["data"]["controls"]
    y = x.select(y_cols)
    x = x.drop(y_cols)
    return y, x


def build_model(
    config: dict,
    x_encoder,
    y_encoder,
    ckpt_path: Optional[Path] = None,
    verbose: bool = False,
) -> LightningModule:
    model_params = config.get("model", {})
    activation = model_params.get("activation", defaults.ACTIVATION)
    normalize = model_params.get("normalize", defaults.NORMALIZE)
    dropout = model_params.get("dropout", defaults.DROPOUT)

    # encoder block to embed labels into vec with hidden size
    labels_encoder_block = ControlsEncoderBlock(
        encoder_types=y_encoder.types(),
        encoder_sizes=y_encoder.sizes(),
        depth=model_params.get("controls_encoder", {}).get(
            "depth", defaults.DEPTH
        ),
        hidden_size=model_params.get("hidden_size", defaults.WIDTH),
        activation=activation,
        normalize=normalize,
        dropout=dropout,
    )

    # encoder and decoder block to process census data
    encoder = CVAEEncoderBlock(
        encoder_types=x_encoder.types(),
        encoder_sizes=x_encoder.sizes(),
        depth=model_params.get("encoder", {}).get("depth", defaults.DEPTH),
        hidden_size=model_params.get("hidden_size", defaults.WIDTH),
        latent_size=model_params.get("latent_size", defaults.LATENT_SIZE),
        activation=activation,
        normalize=normalize,
        dropout=dropout,
    )
    decoder = CVAEDecoderBlock(
        encoder_types=x_encoder.types(),
        encoder_sizes=x_encoder.sizes(),
        depth=model_params.get("decoder", {}).get("depth", defaults.DEPTH),
        hidden_size=model_params.get("hidden_size", defaults.WIDTH),
        latent_size=model_params.get("latent_size", defaults.LATENT_SIZE),
        activation=activation,
        normalize=normalize,
        dropout=dropout,
    )

    model = CVAE(
        embedding_names=x_encoder.names(),
        embedding_types=x_encoder.types(),
        embedding_weights=x_encoder.weights(),
        labels_encoder_block=labels_encoder_block,
        encoder_block=encoder,
        decoder_block=decoder,
        beta=model_params.get("beta", defaults.DEFAULT_BETA),
        lr=model_params.get("lr", defaults.DEFAULT_LR),
        verbose=verbose,
    )
    if ckpt_path:
        model = model.load_from_checkpoint(ckpt_path)
    return model


def build_callbacks(config: dict):
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=config.get("early_stopping", {}).get(
                "patience", defaults.PATIENCE
            ),
            mode="min",
        ),
        ModelCheckpoint(
            monitor="val_loss",
            save_top_k=1,
            mode="min",
            # dirpath=Path(log_dir, "checkpoints"),
            save_weights_only=True,
        ),
    ]


def run_tests(trainer: Trainer, ckpt_path: Optional[str] = None) -> dict:
    if ckpt_path is None:
        ckpt_path = "best"

    losses = {}

    losses["val"] = trainer.validate(
        ckpt_path=ckpt_path, dataloaders=trainer.datamodule.val_dataloader()
    )

    if trainer.datamodule.test_dataloader() is not None:
        losses["test"] = trainer.test(
            ckpt_path=ckpt_path,
            dataloaders=trainer.datamodule.test_dataloader(),
        )

    return losses


def build_gen_loader(config: dict, y_dataset: Tensor) -> DataLoader:
    n = len(y_dataset)
    latent_size = config.get("model", {}).get(
        "latent_size", defaults.LATENT_SIZE
    )
    z_loader = ZDataset(n, latent_size)
    yz_loader = YZDataset(y_dataset, z_loader)
    return DataLoader(yz_loader, **config.get("gen_dataloader", {}))


def generate(
    config: dict, dataloader: DataLoader, trainer: Trainer
) -> Tuple[Tensor, Tensor, Tensor]:
    ys, xs, zs = zip(*trainer.predict(dataloaders=dataloader, ckpt_path="best"))
    ys = concat(ys)
    xs = [concat(x) for x in zip(*xs)]
    zs = concat(zs)
    return ys, xs, zs


def sample(config: dict, xs: Tensor) -> list[Tensor]:
    """Sample from the output distributions.

    Args:
        xs (Tensor): List of output tensors from the model.

    Returns:
        list[Tensor]: Sampled tensors.
    """
    sampler = config.get("generate", {}).get("sample", "argmax")
    if sampler == "argmax":
        return argmax_sampling(xs)
    elif sampler == "multinomial":
        return multinomial_sampling(xs)
    else:
        raise ValueError(f"Unknown sampling method: {sampler}")


def multinomial_sampling(xs: Tensor) -> list[Tensor]:
    sampled = []
    for x in xs:
        if x.dim() == 1:
            sampled.append(x)
        else:
            # check if any values are negative or NaN
            if isnan(x).any():
                sampled.append(x.argmax(dim=-1))
            else:
                sampled.append(multinomial(x, num_samples=1).squeeze(1))
    return stack(sampled, dim=-1)


def argmax_sampling(xs: Tensor) -> list[Tensor]:
    sampled = []
    for x in xs:
        if x.dim() == 1:
            sampled.append(x)
        else:
            sampled.append(argmax(x, dim=-1))
    return stack(sampled, dim=-1)


def evaluate(
    target: pl.DataFrame, synthetic: pl.DataFrame, config: dict
) -> dict:
    """Evaluate the synthetic data against the target data.

    Args:
        target (pl.DataFrame): Target DataFrame.
        synthetic (pl.DataFrame): Synthetic DataFrame.
        config (dict): Configuration dictionary.

    Returns:
        dict: Evaluation metrics.
    """
    yx_binned, synth_binned = binning.bin_continuous(target, synthetic, bins=10)

    density_orders = config.get("evaluate", {}).get("densities", [1, 2, 3])
    homogeneity = config.get("evaluate", {}).get("homogeneity", True)
    incorrectness = config.get("evaluate", {}).get("incorrectness", True)

    metrics = {
        f"density_{o}": density.mean_mean_absolute_error(
            target=yx_binned,
            synthetic=synth_binned,
            order=o,
            controls=config.get("data").get("controls"),
        )
        for o in density_orders
    }
    if homogeneity:
        metrics["homogeneity"] = creativity.simpsons_index(synth_binned)
    if incorrectness:
        metrics["incorrectness"] = correctness.incorrectness(
            yx_binned, synth_binned
        )
    metrics["meta_score"] = sum(list(metrics.values()))

    return metrics
