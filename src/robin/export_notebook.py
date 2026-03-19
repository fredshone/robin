from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat
from nbconvert.exporters import HTMLExporter, MarkdownExporter, WebPDFExporter


def export_notebook(
    notebook_path: Path,
    output_dir: Path,
    stem: str | None = None,
    export_pdf: bool = False,
) -> None:
    notebook = nbformat.read(notebook_path, as_version=4)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = stem or notebook_path.stem

    html_exporter = HTMLExporter()
    html_body, _ = html_exporter.from_notebook_node(notebook)
    (output_dir / f"{base_name}.html").write_text(html_body, encoding="utf-8")

    markdown_exporter = MarkdownExporter()
    markdown_body, resources = markdown_exporter.from_notebook_node(notebook)
    (output_dir / f"{base_name}.md").write_text(markdown_body, encoding="utf-8")

    outputs = resources.get("outputs", {}) if resources else {}
    if outputs:
        assets_dir = output_dir / f"{base_name}_files"
        assets_dir.mkdir(parents=True, exist_ok=True)
        for filename, data in outputs.items():
            asset_path = assets_dir / filename
            asset_path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(data, str):
                asset_path.write_text(data, encoding="utf-8")
            else:
                asset_path.write_bytes(data)

    if export_pdf:
        try:
            pdf_exporter = WebPDFExporter()
            pdf_body, _ = pdf_exporter.from_notebook_node(notebook)
            (output_dir / f"{base_name}.pdf").write_bytes(pdf_body)
        except Exception as exc:
            print(
                "PDF export failed. Install browser dependencies for nbconvert WebPDF "
                "(for example, playwright + chromium), then try again with --pdf.",
                file=sys.stderr,
            )
            print(f"Details: {exc}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a Jupyter notebook to shareable HTML, Markdown, and optional PDF files."
    )
    parser.add_argument("notebook", type=Path, help="Path to .ipynb file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/notebooks"),
        help="Directory where exported files are written",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Output file stem (defaults to notebook filename)",
    )
    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Also export PDF via nbconvert WebPDFExporter",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_notebook(args.notebook, args.output_dir, args.name, args.pdf)
    exported_formats = [".html", ".md"] + ([".pdf"] if args.pdf else [])
    stem = args.name or args.notebook.stem
    formats = ", ".join(f"{stem}{ext}" for ext in exported_formats)
    print(f"Exported {args.notebook} to {args.output_dir} as {formats}")


if __name__ == "__main__":
    main()
