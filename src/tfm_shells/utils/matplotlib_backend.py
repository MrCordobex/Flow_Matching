from __future__ import annotations

import os


def configure_matplotlib_backend(default_backend: str = "Agg") -> None:
    current_backend = os.environ.get("MPLBACKEND", "").strip()
    if not current_backend:
        os.environ["MPLBACKEND"] = default_backend
        return

    # Colab and notebook kernels may inject an inline backend that is not
    # available inside isolated uv/venv script executions.
    if current_backend.startswith("module://matplotlib_inline"):
        os.environ["MPLBACKEND"] = default_backend
