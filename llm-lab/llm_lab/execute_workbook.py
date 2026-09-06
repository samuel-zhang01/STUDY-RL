"""Execute the numerical workbook in a fresh kernel without downloading a model."""
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main() -> None:
    lab = Path(__file__).resolve().parents[1]
    notebook = nbformat.read(lab / "workbook.ipynb", as_version=4)
    nbformat.validate(notebook)
    # Explicit interpreter prevents a globally registered kernel using the wrong venv.
    from jupyter_client import KernelManager
    import sys
    kernel = KernelManager(kernel_name="python3")
    kernel.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    try:
        NotebookClient(notebook, timeout=120, kernel_name="python3", km=kernel,
                       resources={"metadata": {"path": str(lab)}}).execute()
    finally:
        if kernel.has_kernel:
            kernel.shutdown_kernel(now=True)
        kernel.cleanup_resources()
    nbformat.write(notebook, lab / "workbook.executed.ipynb")
    print("Executed every cell; saved llm-lab/workbook.executed.ipynb")


if __name__ == "__main__":
    main()
