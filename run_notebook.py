"""Execute the full experiment with the registered CUDA Jupyter kernel."""
import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", default="industrial-defects-rtx5070")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    path = root / "steel_surface_defect_classification.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)

    def save_progress(**kwargs):
        nbformat.write(notebook, path)
        print(f"Completed cell {kwargs['cell_index']}", flush=True)

    client = NotebookClient(
        notebook, kernel_name=args.kernel, timeout=2400,
        resources={"metadata": {"path": str(root)}},
        on_cell_executed=save_progress,
    )
    try:
        client.execute()
    finally:
        nbformat.write(notebook, path)


if __name__ == "__main__":
    main()
