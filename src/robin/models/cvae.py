from typing import List, Optional

import torch
from pytorch_lightning import LightningModule
from torch import Tensor, exp, nn, ones_like, optim


class CVAE(LightningModule):
    def __init__(
        self,
        embedding_names: list,
        embedding_types: list,
        slot_idxs: list,
        labels_encoder_block: nn.Module,
        encoder_block: nn.Module,
        decoder_block: nn.Module,
        beta: float,
        lr: float,
        token_weights: Optional[dict] = None,
        use_token_weights: bool = False,
        use_weighted_controls: bool = False,
        sampler: Optional[str] = None,
        verbose: bool = False,
    ):
        super().__init__()
        if not (embedding_names and embedding_types):
            raise ValueError("Embedding names and types must be provided")
        if len(embedding_names) != len(embedding_types):
            raise ValueError(
                "Embedding names and types must have the same length"
            )
        self.embedding_names = embedding_names
        self.embedding_types = embedding_types
        self.slot_idxs = slot_idxs

        if use_token_weights:
            self.token_weights = token_weights
        else:
            self.token_weights = [None] * len(embedding_names)
        self.use_weighted_controls = use_weighted_controls

        self.sampler = sampler

        self.labels_encoder_block = labels_encoder_block
        self.encoder_block = encoder_block
        self.decoder_block = decoder_block

        self.beta = beta
        self.lr = lr
        self.verbose = verbose
        self.save_hyperparameters(
            ignore=["labels_encoder_block", "encoder_block", "decoder_block"]
        )
        criterion = []
        for etype, weights in zip(self.embedding_types, self.token_weights):
            if etype == "continuous":
                criterion.append(nn.MSELoss(reduction="none"))
            elif etype == "categorical":
                criterion.append(nn.NLLLoss(weight=weights, reduction="none"))
            elif etype == "decomposed":
                criterion.append(
                    DecomposedLoss(weights=weights, reduction="none")
                )
            else:
                raise ValueError(f"Unknown embedding type: {etype}")

        self.criterion = nn.ModuleList(criterion)

    def forward(
        self, y: Tensor, x: Tensor, target=None, **kwargs
    ) -> List[Tensor]:
        h_y = self.labels_encoder_block(y)
        mu, log_var = self.encode(h_y, x)
        z = self.reparameterize(mu, log_var)
        log_probs_x = self.decode(h_y, z)
        return [log_probs_x, mu, log_var, z]

    def encode(self, hidden_y: Tensor, x: Tensor) -> list[Tensor]:
        return self.encoder_block(hidden_y, x)

    def decode(self, hidden_y: Tensor, z: Tensor, **kwargs) -> List[Tensor]:
        return self.decoder_block(hidden_y, z)

    def loss_function(
        self,
        log_probs: List[Tensor],
        mu: Tensor,
        log_var: Tensor,
        targets: Tensor,
        weights: Tensor,
        **kwargs,
    ) -> dict:
        verbose_metrics = {}
        recons = []

        if self.use_weighted_controls:
            weights = weights / weights.mean()
        else:
            weights = ones_like(weights)

        for name, etype, (i, j), lprobs, criterion in zip(
            self.embedding_names,
            self.embedding_types,
            self.slot_idxs,
            log_probs,
            self.criterion,
        ):
            if etype == "continuous":
                target = targets[:, i:j].squeeze(-1)
                loss = criterion(lprobs, target)
                loss = loss * weights
                loss = loss.mean()
                recons.append(loss)
                verbose_metrics[f"recon_mse_{name}"] = loss
            elif etype == "categorical":
                target = targets[:, i:j].squeeze(-1)
                loss = criterion(lprobs, target.long())
                loss = loss * weights
                loss = loss.mean()
                recons.append(loss)
                verbose_metrics[f"recon_nll_{name}"] = loss
            elif etype == "decomposed":
                target = targets[:, i:j]
                loss = criterion(lprobs, target)
                loss = loss * weights
                loss = loss.mean()
                recons.append(loss)
                verbose_metrics[f"recon_decomposed_{name}"] = loss
            else:
                raise ValueError(f"Unknown encoding for {name}, type: {etype}")

        recon = sum(recons) / len(recons)
        b_recon = (1 - self.beta) * recon

        kld = self.kld(mu, log_var)
        b_kld = self.beta * kld

        loss = b_recon + b_kld

        metrics = {"loss": loss, "kld": b_kld, "recon": b_recon}
        if self.verbose:
            metrics.update(verbose_metrics)

        return metrics

    def kld(self, mu: Tensor, log_var: Tensor) -> Tensor:
        kld = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
        )
        return kld

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return (eps * std) + mu

    def predict(self, y: Tensor, z: Tensor, **kwargs) -> List[Tensor]:
        h_y = self.labels_encoder_block(y)
        log_probs = self.decode(h_y, z, **kwargs)

        if self.sampler is None:
            raise ValueError(
                "Sampler method must be specified in model config for generation."
            )

        sampler = {
            "argmax": torch.argmax,
            "sample": multinomial_sampler,
            "multinomial": multinomial_sampler,
        }.get(self.sampler)

        if sampler is None:
            raise ValueError(f"Unknown sampling method: {self.sampler}")

        preds = []
        for name, etype, lprobs in zip(
            self.embedding_names, self.embedding_types, log_probs
        ):
            if etype == "continuous":
                preds.append(lprobs.unsqueeze(-1))
            elif etype == "categorical":
                preds.append(sampler(lprobs, dim=1).unsqueeze(-1))
            elif etype == "decomposed":
                v, k = lprobs[:, 0], lprobs[:, 1:]
                components = sampler(k, dim=1)
                preds.append(torch.stack([v, components], dim=1))
            else:
                raise ValueError(f"Unknown encoding for {name}, type: {etype}")

        preds = torch.cat(preds, dim=1)

        return y, preds, z

    # def infer(self, y: Tensor, x: Tensor, **kwargs) -> Tensor:
    #     log_probs_x, _, _, z = self.forward(y, x, **kwargs)
    #     prob_samples = torch.exp(log_probs_x)
    #     return prob_samples, z

    def training_step(self, batch, batch_idx):
        (y, yw), (x, _) = batch
        log_probs, mu, log_var, _ = self.forward(y, x)
        train_losses = self.loss_function(
            log_probs=log_probs, mu=mu, log_var=log_var, targets=x, weights=yw
        )
        self.log_dict(
            {key: val.item() for key, val in train_losses.items()},
            sync_dist=True,
        )
        return train_losses["loss"]

    def validation_step(self, batch, batch_idx):
        (y, yw), (x, _) = batch
        log_probs, mu, log_var, _ = self.forward(y, x)
        loss = self.loss_function(
            log_probs=log_probs, mu=mu, log_var=log_var, targets=x, weights=yw
        )
        self.log_dict(
            {f"val_{key}": val.item() for key, val in loss.items()},
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def test_step(self, batch):
        (y, yw), (x, _) = batch
        log_probs, mu, log_var, _ = self.forward(y, x)
        loss = self.loss_function(
            log_probs=log_probs, mu=mu, log_var=log_var, targets=x, weights=yw
        )
        self.log_dict({f"test_{key}": val.item() for key, val in loss.items()})

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        return [optimizer], [scheduler]

    def predict_step(self, batch):
        (y, _), z = batch
        return self.predict(y, z)


class DecomposedLoss(nn.Module):
    # todo:option to replace mse with gaussian negative log likelihood
    def __init__(
        self, weights: Optional[Tensor] = None, reduction: str = "none"
    ):
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction=reduction)
        self.nll_loss = nn.NLLLoss(weight=weights, reduction=reduction)

    def forward(self, log_probs: List[Tensor], target: Tensor) -> Tensor:
        continuous_target = target[:, 0]
        categorical_target = target[:, 1].long()
        continuous_preds = log_probs[:, 0]
        categorical_preds = log_probs[:, 1:]

        mse = self.mse_loss(continuous_preds, continuous_target)
        nll = self.nll_loss(categorical_preds, categorical_target)

        return mse + nll


def multinomial_sampler(log_probs: Tensor, dim: int) -> Tensor:
    return torch.multinomial(exp(log_probs), num_samples=1).squeeze(-1)
