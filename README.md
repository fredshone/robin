# Robin

Robin generates synthetic census populations using Conditional Variational Autoencoders (CVAE). It encodes demographic data, trains a CVAE conditioned on control variables (e.g., sex, age group, region), and decodes synthetic populations that respect those constraints.

## Installation

```bash
pip install uv
uv sync
```

## Quick Start

```bash
robin run configs/config.yaml
```

This trains a model using the settings in the YAML config and writes evaluation outputs to the configured log directory.

## Configuration

YAML configs under `configs/` drive the full pipeline. Key sections:

```yaml
seed: 1234

data:
  train_path: path/to/data.csv
  columns:
    sex: sex                      # column_name: alias
    resident_age_7d: age_group
    region: region
    # ... additional demographic columns
  controls:
    - sex                         # conditioning variables
    - age_group
    - region

datamodule:
  val_split: 0.1
  train_batch_size: 1024

model:
  latent_size: 20
  beta: 1
  lr: 0.1
  hidden_size: 64
  controls_encoder:
    depth: 4
  encoder:
    depth: 4
  decoder:
    depth: 4

trainer:
  min_epochs: 5
  max_epochs: 100
  early_stopping:
    patience: 10

evaluate:
  densities: [1, 2, 3]           # marginal order for correctness eval
```

Column encoders are inferred automatically by type. Supported encoders: `MinMax`, `StandardScaler`, `GMM` (Gaussian mixture for continuous columns), `Meta` (auto-selecting GMM, see below), `Categorical`.

### Meta encoder

`MetaDecomposer` (`continuous_encoding: meta`) is a drop-in replacement for `GMMEncoder` that automatically selects the best pre-transformation for each column. At fit time it tries a ranked set of transforms (identity, min-max, standard scaler, log), fits a Bayesian GMM on the result of each, and keeps the one with the highest log-likelihood. This is useful when columns have different distributional shapes (skewed, heavy-tailed, bounded) and you don't want to tune the transform per column by hand.

```yaml
encoder:
  continuous_encoding: meta
  max_components: 10
```

## CLI

```bash
robin run configs/config.yaml          # train and evaluate
robin sweep configs/sweep.yaml         # W&B hyperparameter sweep
robin tune configs/tune_demo.yaml      # Optuna tuning
```

## Architecture

Data flows through three stages: **encode → train/generate → decode**.

**Encoding** (`robin/encoders/`)  
`TableEncoder` orchestrates per-column encoders and produces flat tensor batches. Column encoders handle numeric scaling (`MinMaxEncoder`, `StandardScaler`), categorical tokenisation (`CategoricalTokeniser`), and Gaussian mixture decomposition for continuous columns (`GMMEncoder`).

**Model** (`robin/models/`)  
`CVAE` is a PyTorch Lightning module. `ControlsEncoderBlock` embeds the conditioning variables; `CVAEEncoderBlock` compresses features to a latent vector; `CVAEDecoderBlock` reconstructs features from the latent vector and conditioning embeddings.

**Training / Generation** (`robin/runners/`, `robin/dataloaders/`)  
`DataModule` handles train/val/test splits. `z_loader.py` samples latent vectors for generation conditioned on control variable marginals.

**Evaluation** (`robin/eval/`)  
- `correctness.py`: marginal distribution accuracy (1st/2nd/3rd order)
- `density.py`: distribution similarity (KL divergence)
- `creativity.py`: novelty of synthetic value combinations

## Notebook Export

To export the evaluation notebook to HTML and Markdown:

```bash
uv run robin-export-notebook demos/evaluate.ipynb --output-dir reports/notebooks --name evaluate
```

Add `--pdf` for PDF output. If PDF export fails on first run, install browser support:

```bash
uv pip install "nbconvert[webpdf]" playwright
uv run playwright install chromium
```

## Development

```bash
pytest                              # full test suite
pytest tests/                       # unit tests only
ruff check src/                     # lint
black src/                          # format
```

## License

MIT
