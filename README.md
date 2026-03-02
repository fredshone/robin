Notes:

Nomis has annual projections for ages and genders by geographies.

To export the notebook into shareable formats (HTML + Markdown), run:

`uv run robin-export-notebook demos/evaluate.ipynb --output-dir reports/notebooks --name evaluate`

To also export PDF:

`uv run robin-export-notebook demos/evaluate.ipynb --output-dir reports/notebooks --name evaluate --pdf`

If PDF export fails the first time, install browser support once:

- `uv pip install "nbconvert[webpdf]" playwright`
- `uv run playwright install chromium`

This writes:

- `reports/notebooks/evaluate.html`
- `reports/notebooks/evaluate.md`
- `reports/notebooks/evaluate.pdf` (when `--pdf` succeeds)
- `reports/notebooks/evaluate_files/` (embedded assets like images)