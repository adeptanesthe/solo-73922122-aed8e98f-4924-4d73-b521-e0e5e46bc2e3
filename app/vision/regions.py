"""Classical-CV key-region detection for fault screenshots.

Detects visually salient / error-indicating regions without any learned model:
  * red-dominant areas   -> typical error banners / failure buttons
  * amber-dominant areas -> typical warning strips
  * high-contrast boxed areas -> dialogs / panels that stand out from background

Works on a downscaled grid with BFS connected components, then maps boxes back
to full resolution. Deterministic and fast (numpy only).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

_CELL = 6  # grid cell size in pixels (analysis resolution = image / _CELL)
_MIN_CELLS = 6  # ignore components smaller than this many cells


@dataclass
class Region:
    kind: str  # red_alert | amber_warning | contrast_block
    x: int
    y: int
    width: int
    height: int
    score: float
    notes: list[str] = field(default_factory=list)

    @property
    def area(self) -> int:
        return self.width * self.height


def _masks(arr: np.ndarray) -> dict[str, np.ndarray]:
    r = arr[..., 0].astype(np.int16)
    g = arr[..., 1].astype(np.int16)
    b = arr[..., 2].astype(np.int16)
    red = (r > 140) & (g < 130) & (r - g > 60) & (r - b > 60)
    amber = (r > 200) & (g >= 120) & (g <= 200) & (b < 80) & (r - b > 120)
    return {"red_alert": red, "amber_warning": amber}


def _contrast_mask(arr: np.ndarray) -> np.ndarray:
    """Cells whose local luminance std is high -> boxed panels/dialogs/edges."""
    lum = arr.astype(np.float32).mean(axis=2)
    h, w = lum.shape
    gh, gw = max(h // _CELL, 1), max(w // _CELL, 1)
    grid = np.zeros((gh, gw), dtype=bool)
    # global background = most common brightness band; blocks deviate from it
    for gy in range(gh):
        for gx in range(gw):
            cell = lum[gy * _CELL : (gy + 1) * _CELL, gx * _CELL : (gx + 1) * _CELL]
            if cell.size and cell.std() > 55.0:
                grid[gy, gx] = True
    return grid


def _to_grid(mask: np.ndarray, gh: int, gw: int) -> np.ndarray:
    h, w = mask.shape
    grid = np.zeros((gh, gw), dtype=bool)
    for gy in range(gh):
        for gx in range(gw):
            cell = mask[gy * _CELL : (gy + 1) * _CELL, gx * _CELL : (gx + 1) * _CELL]
            if cell.size and cell.mean() > 0.35:
                grid[gy, gx] = True
    return grid


def _components(grid: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    """BFS connected components on a boolean grid.

    Returns list of (gy0, gx0, gy1, gx1, size) with exclusive max bounds.
    """
    seen = np.zeros_like(grid, dtype=bool)
    out: list[tuple[int, int, int, int, int]] = []
    gh, gw = grid.shape
    for sy in range(gh):
        for sx in range(gw):
            if not grid[sy, sx] or seen[sy, sx]:
                continue
            q = deque([(sy, sx)])
            seen[sy, sx] = True
            ys, xs = [sy], [sx]
            while q:
                y, x = q.popleft()
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < gh and 0 <= nx < gw and grid[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        q.append((ny, nx))
                        ys.append(ny)
                        xs.append(nx)
            if len(ys) >= _MIN_CELLS:
                out.append((min(ys), min(xs), max(ys) + 1, max(xs) + 1, len(ys)))
    return out


def _contained(a: Region, b: Region) -> float:
    """Fraction of region `a` covered by region `b`."""
    x0, y0 = max(a.x, b.x), max(a.y, b.y)
    x1, y1 = min(a.x + a.width, b.x + b.width), min(a.y + a.height, b.y + b.height)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return ((x1 - x0) * (y1 - y0)) / a.area


def detect_regions(image: Image.Image, max_regions: int = 8) -> list[Region]:
    """Detect key regions; returns boxes sorted by score, highest first."""
    img = image.convert("RGB")
    w, h = img.size
    arr = np.asarray(img)
    gh, gw = max(h // _CELL, 1), max(w // _CELL, 1)

    regions: list[Region] = []
    for kind, mask in _masks(arr).items():
        grid = _to_grid(mask, gh, gw)
        for gy0, gx0, gy1, gx1, size in _components(grid):
            frac = size / (gh * gw)
            score = 0.55 + 0.45 * min(frac * 12.0, 1.0)  # bigger -> higher, capped
            regions.append(
                Region(
                    kind=kind,
                    x=gx0 * _CELL,
                    y=gy0 * _CELL,
                    width=min((gx1 - gx0) * _CELL, w - gx0 * _CELL),
                    height=min((gy1 - gy0) * _CELL, h - gy0 * _CELL),
                    score=round(min(score, 0.98), 3),
                    notes=[f"{kind} covers {frac:.1%} of image"],
                )
            )

    # contrast blocks: keep only sizable ones, score lower than explicit alerts
    cgrid = _contrast_mask(arr)
    for gy0, gx0, gy1, gx1, size in _components(cgrid):
        frac = size / (gh * gw)
        if frac < 0.01:  # skip tiny edge noise
            continue
        bw, bh = (gx1 - gx0) * _CELL, (gy1 - gy0) * _CELL
        if min(bw, bh) < 2 * _CELL:  # skip thin edge strips
            continue
        regions.append(
            Region(
                kind="contrast_block",
                x=gx0 * _CELL,
                y=gy0 * _CELL,
                width=min((gx1 - gx0) * _CELL, w - gx0 * _CELL),
                height=min((gy1 - gy0) * _CELL, h - gy0 * _CELL),
                score=round(0.35 + 0.3 * min(frac * 8.0, 1.0), 3),
                notes=[f"high-contrast block covers {frac:.1%} of image"],
            )
        )

    # drop regions mostly contained inside a higher-scored one
    regions.sort(key=lambda r: r.score, reverse=True)
    kept: list[Region] = []
    for r in regions:
        if any(_contained(r, k) > 0.8 for k in kept):
            continue
        kept.append(r)
        if len(kept) >= max_regions:
            break
    return kept
