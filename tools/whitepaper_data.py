"""Write the whitepaper's data files from the docs' own sources.

``just whitepaper`` runs it before compiling the paper. It writes, under
``whitepaper/data``:

- ``front.json``: the 3D bend's history, parameters and results
  (from ``docs/_data/front-page.json``, built by ``tools/front_page_data.py``);
- ``snapshot-<iteration>.png``: the four design snapshots of the front page,
  the guides plus the design square's density in ink at alpha 0.88 v, as
  ``docs/_static/front/front.js`` draws them;
- ``caps.json``: the gradient matrix and the raises list, derived from the
  verification tests by ``docs/_ext/gfem_capabilities.py``.

Nothing here is computed: every number comes from a docs data file or a test.
"""

from __future__ import annotations

import base64
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "whitepaper" / "data"
INK = (38, 38, 38)  # --ink-field in docs/_static/gfem.css
SCALE = 8  # nearest-neighbour upscale: the snapshots stay pixelated


def snapshots(front: dict) -> list[dict]:
    """Render each snapshot as an RGBA PNG of the whole box, y up."""
    x0, x1, _, y1 = front["extent_um"]
    d, w = front["design_um"], front["guide_um"]
    snap = front["snapshots"]
    ny, nx = snap["shape"]
    pitch = 2 * d / nx
    n = round((x1 - x0) / pitch)
    xs = x0 + (np.arange(n) + 0.5) * pitch
    ys = y1 - (np.arange(n) + 0.5) * pitch  # rows from max y down
    X, Y = np.meshgrid(xs, ys)
    guides = ((-d > X) & (np.abs(Y) < w)) | ((-d > Y) & (np.abs(X) < w))
    i0 = round((y1 - d) / pitch)
    j0 = round((-d - x0) / pitch)
    out = []
    for it, t, b64 in zip(snap["iterations"], snap["t"], snap["density"], strict=True):
        rho = np.frombuffer(base64.b64decode(b64), dtype=np.uint8).reshape(ny, nx) / 255
        alpha = np.where(guides, 0.88, 0.0)
        alpha[i0 : i0 + ny, j0 : j0 + nx] = 0.88 * rho
        rgba = np.zeros((n, n, 4), dtype=np.uint8)
        rgba[..., :3] = INK
        rgba[..., 3] = np.round(255 * alpha).astype(np.uint8)
        size = (n * SCALE, n * SCALE)
        img = Image.fromarray(rgba, "RGBA").resize(size, Image.NEAREST)
        name = f"snapshot-{it}.png"
        img.save(OUT / name)
        out.append({"iteration": it, "t": t, "file": name})
    return out


def capabilities() -> dict:
    """The gradient matrix and the raises list, as the docs build derives them."""
    sys.path.insert(0, str(REPO / "docs" / "_ext"))
    import gfem_capabilities

    caps = gfem_capabilities.derive(REPO)
    keep = ("columns", "rows", "counts", "provenance", "raises")
    return json.loads(json.dumps({k: caps[k] for k in keep}, default=str))


def main() -> None:
    """Write every data file the whitepaper reads."""
    OUT.mkdir(parents=True, exist_ok=True)
    front = json.loads((REPO / "docs/_data/front-page.json").read_text())
    data = {
        "source": front["source"],
        "history": front["history"],
        "params": front["params"],
        "results": front["results"],
        "extent_um": front["extent_um"],
        "design_um": front["design_um"],
        "snapshots": snapshots(front),
    }
    (OUT / "front.json").write_text(json.dumps(data, indent=1))
    (OUT / "caps.json").write_text(json.dumps(capabilities(), indent=1))
    poster = REPO / "docs/_static/front/bend3d-optimization-light.png"
    shutil.copy(poster, OUT / "bend3d-final.png")
    for logo in ("gfem-lockup-light.svg", "gfem-mark-light.svg"):
        shutil.copy(REPO / "docs/_static/logo" / logo, OUT / logo)
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
