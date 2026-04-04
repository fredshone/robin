from torch.utils.data import DataLoader

from robin.encoders.table_datasets import ZDataset


def build_latent_dataloader(
    num_samples, latent_dim: int, batch_size: int = 1024, num_workers: int = 4
):
    return DataLoader(
        ZDataset(num_samples, latent_dim),
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
