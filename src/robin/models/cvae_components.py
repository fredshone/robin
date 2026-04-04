from typing import Tuple

from torch import Tensor, cat, nn, stack


class ControlsEncoderBlock(nn.Module):
    def __init__(
        self,
        encoder_types: list,
        slot_idxs: list,
        encoder_sizes: list,
        depth: int,
        hidden_size: int,
        skips: bool = True,
        normalize: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.skips = skips
        self.embed = Embedder(
            encoder_types, slot_idxs, encoder_sizes, hidden_size
        )
        self.ff = FFBlock(
            hidden_size, hidden_size, depth, hidden_size, dropout=dropout
        )
        self.normalize = normalize
        self.norm = nn.BatchNorm1d(hidden_size)

    def forward(self, y: Tensor) -> Tensor:
        e = self.embed(y)
        h = self.ff(e)
        if self.skips:
            h = e + h
        if self.normalize:
            h = self.norm(h)
        return h


class CVAEEncoderBlock(nn.Module):
    def __init__(
        self,
        encoder_types: list,
        slot_idxs: list,
        encoder_sizes: list,
        depth: int,
        hidden_size: int,
        latent_size: int,
        skips: bool = True,
        normalize: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.skips = skips
        self.normalize = normalize
        self.embed = Embedder(
            encoder_types, slot_idxs, encoder_sizes, hidden_size
        )
        self.ff = FFBlock(
            hidden_size, hidden_size, depth, hidden_size, dropout=dropout
        )
        self.norm = nn.BatchNorm1d(hidden_size)
        self.fc_mu = nn.Linear(hidden_size, latent_size)
        self.fc_var = nn.Linear(hidden_size, latent_size)

    def forward(self, hidden_y: Tensor, x: Tensor) -> Tuple[Tensor, Tensor]:
        e = self.embed(x)
        e = e + hidden_y
        h = self.ff(e)
        if self.normalize:
            h = self.norm(h)
        if self.skips:
            h = e + h
        mu = self.fc_mu(h)
        var = self.fc_var(h)
        return mu, var


class CVAEDecoderBlock(nn.Module):
    def __init__(
        self,
        encoder_types: list,
        encoder_sizes: list,
        depth,
        hidden_size,
        latent_size,
        skips: bool = True,
        normalize: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.skips = skips
        self.normalize = normalize
        self.resize = nn.Linear(latent_size, hidden_size)
        self.zff = FFBlock(
            latent_size, hidden_size, depth, hidden_size, dropout=dropout
        )
        self.norm = nn.BatchNorm1d(hidden_size)
        embeds = []
        for type, size in zip(encoder_types, encoder_sizes):
            if type == "continuous":
                embeds.append(
                    nn.Sequential(nn.Linear(hidden_size, 1), nn.Tanh())
                )
            elif type == "categorical":
                embeds.append(
                    nn.Sequential(
                        nn.Linear(hidden_size, size), nn.LogSoftmax(dim=-1)
                    )
                )
            elif type == "decomposed":
                embeds.append(DecomposedDecoder(hidden_size, size))
            else:
                raise ValueError(f"Unknown encoder type: {type}")

        self.embeds = nn.ModuleList(embeds)

    def forward(self, hidden_y: Tensor, z: Tensor) -> Tensor:
        h = self.zff(z)
        if self.normalize:
            h = self.norm(h)
        if self.skips:
            z = self.resize(z)
            h = h + z
        h = h + hidden_y
        xs = [embed(h).squeeze(1) for embed in self.embeds]
        return xs


class Embedder(nn.Module):
    def __init__(
        self,
        encoder_types: list,
        slot_idxs: list,
        encoder_sizes: list,
        embed_size,
    ):
        super().__init__()
        self.encoder_types = encoder_types
        self.slot_idxs = slot_idxs
        embeds = []

        for encoding_type, size in zip(encoder_types, encoder_sizes):
            if encoding_type == "categorical":
                embeds.append(CatEmbedding(size, embed_size))

            elif encoding_type == "continuous":
                embeds.append(NumericEmbedding(embed_size))

            elif encoding_type == "decomposed":
                embeds.append(DecomposedEmbedding(size, embed_size))

        self.embeds = nn.ModuleList(embeds)

    def forward(self, x: Tensor) -> Tensor:
        # TODO: need to remove loop for speed
        xs = []
        for embed, (i, j) in zip(self.embeds, self.slot_idxs):
            col = x[:, i:j]
            # TODO: need to separate x_cat and x_cont
            embedded = embed(col)
            xs.append(embedded)
        # consider splitting categorical and continuous in future
        xs = stack(xs, dim=-1)
        xs = xs.sum(dim=-1)
        return xs


class Noop(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


class FFBlock(nn.Module):
    # check about removing extra bias
    # add skipping connections
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        depth: int,
        output_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        if depth < 0:
            raise ValueError("depth must be non-negative")
        elif depth == 0:
            if input_size == output_size:
                block = [Noop()]
            else:
                raise ValueError(
                    "input_size must equal output_size when depth is 0"
                )
        else:
            block = [nn.Linear(input_size, hidden_size), nn.LeakyReLU()]
            for _ in range(depth - 1):
                block.extend(
                    [nn.Linear(hidden_size, hidden_size), nn.LeakyReLU()]
                )
            block.append(nn.Linear(hidden_size, output_size))
        if dropout > 0:
            block.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*block)

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class CatEmbedding(nn.Module):
    def __init__(self, num_embeddings, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, hidden_size)

    def forward(self, x):
        return self.embedding(x.long()).squeeze(1)


class NumericEmbedding(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(1, hidden_size)

    def forward(self, x):
        return self.fc(x)


class DecomposedEmbedding(nn.Module):
    def __init__(self, num_embeddings, hidden_size):
        super().__init__()
        self.fc = nn.Linear(1, hidden_size)
        self.embedding = nn.Embedding(num_embeddings, hidden_size)

    def forward(self, x):
        v, k = x[:, 0:1], x[:, 1:2].long()
        return self.fc(v) + self.embedding(k).squeeze(1)


class DecomposedDecoder(nn.Module):
    def __init__(self, hidden_size, size):
        super().__init__()
        self.value_net = nn.Sequential(nn.Linear(hidden_size, 1), nn.Tanh())
        self.component_net = nn.Sequential(
            nn.Linear(hidden_size, size), nn.LogSoftmax(dim=-1)
        )

    def forward(self, x):
        v = self.value_net(x)
        k = self.component_net(x)
        return cat([v, k], dim=-1)
