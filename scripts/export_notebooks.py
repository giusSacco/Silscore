"""Generate executed notebooks with embedded figures from the paired Python files."""
from pathlib import Path
import re
import sys

import nbformat
from nbclient import NotebookClient
from jupyter_client import AsyncKernelManager
from jupyter_client.kernelspec import KernelSpecManager


ROOT = Path(__file__).resolve().parents[1]


def from_script(script: Path):
    """Keep the script's code and cell order, including its module prelude."""
    chunks = re.split(r"(?m)^# %%[^\n]*\n", script.read_text())
    if len(chunks) > 1:
        chunks[1] = chunks[0] + chunks[1]
        chunks = chunks[1:]
    title = script.stem.partition("_")[2].replace("_", " ").capitalize()
    intro = (
        f"# {title}\n\n"
        "Run all cells with the `silscore` environment; the included caches are sufficient.\n\n"
        f"Generated from [{script.name}]({script.name})."
    )
    cells = [nbformat.v4.new_markdown_cell(intro)]
    cells += [nbformat.v4.new_code_cell(code.strip()) for code in chunks if code.strip()]
    for index, cell in enumerate(cells):
        cell.id = f"cell-{index}"
    return nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
    })


def main() -> int:
    """Execute each notebook in a new kernel and save its rendered outputs."""
    import matplotlib.font_manager  # Warm the font cache before capturing cell output.

    specs = KernelSpecManager(kernel_dirs=[str(Path(sys.prefix) / "share/jupyter/kernels")])
    for script in sorted((ROOT / "notebooks").glob("nb*.py")):
        notebook = from_script(script)
        kernel = AsyncKernelManager(kernel_name="python3", kernel_spec_manager=specs)
        NotebookClient(
            notebook, km=kernel, timeout=180, store_widget_state=False,
            resources={"metadata": {"path": str(script.parent)}},
        ).execute(cleanup_kc=True)
        for cell in notebook.cells:
            cell.metadata.clear()
            for output in cell.get("outputs", []):
                if output.output_type == "stream":
                    output.text = output.text.replace(str(ROOT), ".")
        if not any("image/png" in output.get("data", {})
                   for cell in notebook.cells for output in cell.get("outputs", [])):
            raise RuntimeError(f"{script.name}: no embedded figures")
        nbformat.validate(notebook)
        output = script.with_suffix(".ipynb")
        nbformat.write(notebook, output)
        print(f"Wrote {output.relative_to(ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
