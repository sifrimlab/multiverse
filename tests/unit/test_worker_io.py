from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

SDK_PATH = Path(__file__).resolve().parents[2] / "sdk" / "mvr-worker"
if str(SDK_PATH) not in sys.path:
    sys.path.insert(0, str(SDK_PATH))


def test_save_umap_uses_png_temp_file_and_explicit_format(tmp_path, monkeypatch):
    class FakeAnnData:
        def __init__(self, obs=None, **_kwargs):
            self.obs = obs
            self.obsm = {}

    anndata = types.SimpleNamespace(
        AnnData=FakeAnnData,
        concat=lambda *_args, **_kwargs: None,
    )
    mudata = types.SimpleNamespace(
        read_h5mu=lambda _path: None,
        MuData=lambda _data: None,
    )
    h5py = types.SimpleNamespace(File=lambda *_args, **_kwargs: None)

    monkeypatch.setitem(sys.modules, "anndata", anndata)
    monkeypatch.setitem(sys.modules, "mudata", mudata)
    monkeypatch.setitem(sys.modules, "h5py", h5py)

    saved = {}

    matplotlib = types.ModuleType("matplotlib")
    matplotlib.__path__ = []
    matplotlib.use = lambda _backend: None

    pyplot = types.ModuleType("matplotlib.pyplot")

    def savefig(path, **kwargs):
        saved["path"] = Path(path)
        saved["kwargs"] = kwargs
        Path(path).write_bytes(b"png")

    pyplot.savefig = savefig
    pyplot.close = lambda: None
    matplotlib.pyplot = pyplot

    scanpy = types.SimpleNamespace(
        pp=types.SimpleNamespace(neighbors=lambda *_args, **_kwargs: None),
        tl=types.SimpleNamespace(umap=lambda *_args, **_kwargs: None),
        pl=types.SimpleNamespace(umap=lambda *_args, **_kwargs: None),
    )

    monkeypatch.setitem(sys.modules, "matplotlib", matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", pyplot)
    monkeypatch.setitem(sys.modules, "scanpy", scanpy)

    from mvr_worker.io import save_umap

    obs = pd.DataFrame({"cell_type": ["a", "b", "a"]}, index=["c1", "c2", "c3"])
    latent = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])

    result = save_umap(latent, obs, str(tmp_path))

    assert result == str(tmp_path / "umap.png")
    assert (tmp_path / "umap.png").read_bytes() == b"png"
    assert saved["path"].suffix == ".png"
    assert saved["path"].name != "umap.png.tmp"
    assert saved["kwargs"]["format"] == "png"
    assert not list(tmp_path.glob(".umap-*.png"))
