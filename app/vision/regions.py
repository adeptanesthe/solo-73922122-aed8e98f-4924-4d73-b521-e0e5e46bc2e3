"""Classical-CV key-region detection for fault screenshots.

Detects visually salient / error-indicating regions without any learned model:
  * red-dominant areas   -> typical error banners / failure buttons
  * amber-dominant areas -> typical warning strips
  * high-contrast boxed areas -> dialogs / panels that stand out from background

The image is processed in fixed-size tiles and the per-pixel colour masks are
never materialised for the whole image: each tile is reduced straight to the
compact cell grid (image / ``_CELL``).  Connected components are labelled per
tile with a union-find that merges labels across shared tile borders, so no
full-resolution boolean/luminance arrays are kept alive.  This keeps peak
memory flat for large screenshots while producing byte-identical regions to
the previous whole-image implementation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

_CELL = 6  # grid cell size in pixels (analysis resolution = image / _CELL)
_MIN_CELLS = 6  # ignore components smaller than this many cells
_TILE_CELLS = 128  # tile size in grid cells -> at most 768x768 px analysed at once
_CELL_COVERAGE = 0.35  # fraction of a cell's pixels that must match the mask
_CONTRAST_STD = 55.0  # per-cell luminance std that marks a high-contrast cell


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


def _pixel_masks(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel red/amber boolean masks for one tile (or a tiny image)."""
    r = arr[..., 0].astype(np.int16)
    g = arr[..., 1].astype(np.int16)
    b = arr[..., 2].astype(np.int16)
    red = (r > 140) & (g < 130) & (r - g > 60) & (r - b > 60)
    amber = (r > 200) & (g >= 120) & (g <= 200) & (b < 80) & (r - b > 120)
    return red, amber


def _cell_counts(mask: np.ndarray, th: int, tw: int) -> np.ndarray:
    """Count matching pixels per _CELL x _CELL cell of a tile."""
    return mask.reshape(th, _CELL, tw, _CELL).sum(axis=(1, 3), dtype=np.int32)


def _cell_std(lum: np.ndarray, th: int, tw: int) -> np.ndarray:
    """Per-cell luminance std.

    Cells are copied into a contiguous (n, _CELL, _CELL) array so the reduction
    order matches a per-cell ``cell.std()`` exactly (a bare ``std(axis=(1, 3))``
    on the strided 4-D view can differ in the last float bits).
    """
    cells = np.ascontiguousarray(
        lum.reshape(th, _CELL, tw, _CELL).transpose(0, 2, 1, 3).reshape(th * tw, _CELL, _CELL)
    )
    return cells.std(axis=(1, 2)).reshape(th, tw)


def _build_grids_small(img: Image.Image, gh: int, gw: int) -> dict[str, np.ndarray]:
    """Exact legacy path for images smaller than one cell in a dimension.

    Only then does a grid cell cover fewer than _CELL*_CELL pixels; the tiled
    path assumes full cells.  Such images are tiny, so the simple loop is fine.
    """
    arr = np.asarray(img)
    lum = arr.astype(np.float32).mean(axis=2)
    red, amber = _pixel_masks(arr)
    grids = {kind: np.zeros((gh, gw), dtype=bool) for kind in ("red_alert", "amber_warning", "contrast_block")}
    for kind, mask in (("red_alert", red), ("amber_warning", amber)):
        for gy in range(gh):
            for gx in range(gw):
                cell = mask[gy * _CELL : (gy + 1) * _CELL, gx * _CELL : (gx + 1) * _CELL]
                if cell.size and cell.mean() > _CELL_COVERAGE:
                    grids[kind][gy, gx] = True
    for gy in range(gh):
        for gx in range(gw):
            cell = lum[gy * _CELL : (gy + 1) * _CELL, gx * _CELL : (gx + 1) * _CELL]
            if cell.size and cell.std() > _CONTRAST_STD:
                grids["contrast_block"][gy, gx] = True
    return grids


def _build_grids(img: Image.Image, gh: int, gw: int) -> dict[str, np.ndarray]:
    """Reduce the image to the red/amber/contrast cell grids, tile by tile."""
    w, h = img.size
    if h < _CELL or w < _CELL:
        return _build_grids_small(img, gh, gw)

    grids = {kind: np.zeros((gh, gw), dtype=bool) for kind in ("red_alert", "amber_warning", "contrast_block")}
    min_count = _CELL_COVERAGE * (_CELL * _CELL)
    for cy in range(0, gh, _TILE_CELLS):
        th = min(_TILE_CELLS, gh - cy)
        for cx in range(0, gw, _TILE_CELLS):
            tw = min(_TILE_CELLS, gw - cx)
            box = (cx * _CELL, cy * _CELL, (cx + tw) * _CELL, (cy + th) * _CELL)
            tile = np.asarray(img.crop(box))
            red, amber = _pixel_masks(tile)
            grids["red_alert"][cy : cy + th, cx : cx + tw] = _cell_counts(red, th, tw) > min_count
            grids["amber_warning"][cy : cy + th, cx : cx + tw] = _cell_counts(amber, th, tw) > min_count
            lum = tile.astype(np.float32).mean(axis=2)
            grids["contrast_block"][cy : cy + th, cx : cx + tw] = _cell_std(lum, th, tw) > _CONTRAST_STD
    return grids


class _DSU:
    """Union-find over per-tile component labels.

    Each node stores its component's cell-count, bounding box and the first
    cell in row-major scan order, which is the order the previous full-image
    BFS discovered components in.  Merging keeps all of these consistent, so a
    component spanning tiles ends up with exactly the legacy bbox/size/order.
    """

    __slots__ = ("parent", "rank", "size", "y0", "x0", "y1", "x1", "first")

    def __init__(self) -> None:
        self.parent: list[int] = []
        self.rank: list[int] = []
        self.size: list[int] = []
        self.y0: list[int] = []
        self.x0: list[int] = []
        self.y1: list[int] = []
        self.x1: list[int] = []
        self.first: list[tuple[int, int]] = []

    def add(self, bbox: tuple[int, int, int, int], first: tuple[int, int], size: int) -> int:
        node = len(self.parent)
        self.parent.append(node)
        self.rank.append(0)
        self.size.append(size)
        y0, x0, y1, x1 = bbox
        self.y0.append(y0)
        self.x0.append(x0)
        self.y1.append(y1)
        self.x1.append(x1)
        self.first.append(first)
        return node

    def find(self, x: int) -> int:
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        self.size[ra] += self.size[rb]
        self.y0[ra] = min(self.y0[ra], self.y0[rb])
        self.x0[ra] = min(self.x0[ra], self.x0[rb])
        self.y1[ra] = max(self.y1[ra], self.y1[rb])
        self.x1[ra] = max(self.x1[ra], self.x1[rb])
        if self.first[rb] < self.first[ra]:
            self.first[ra] = self.first[rb]

    def components(self, min_size: int) -> list[tuple[int, int, int, int, int]]:
        """(gy0, gx0, gy1, gx1, size), exclusive max bounds, legacy scan order.

        Ordering by the component's first cell in row-major scan order matches
        the discovery order of the previous whole-image BFS, which keeps region
        ordering (and therefore evidence numbering) stable.
        """
        out: list[tuple[int, int, int, int, int]] = []
        for node in range(len(self.parent)):
            if self.find(node) != node or self.size[node] < min_size:
                continue
            out.append((self.y0[node], self.x0[node], self.y1[node] + 1, self.x1[node] + 1, self.size[node], self.first[node]))
        out.sort(key=lambda c: c[5])
        return [(y0, x0, y1, x1, size) for y0, x0, y1, x1, size, _ in out]


def _label_block(
    block: np.ndarray,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]], list[tuple[int, int]], list[int]]:
    """4-connected component labels of one grid block (scan-order seeds)."""
    th, tw = block.shape
    labels = np.zeros((th, tw), dtype=np.int32)
    bboxes: list[tuple[int, int, int, int]] = []
    firsts: list[tuple[int, int]] = []
    sizes: list[int] = []
    for y in range(th):
        row = block[y]
        if not row.any():
            continue
        for x in np.flatnonzero(row):
            x = int(x)
            if labels[y, x]:
                continue
            n = len(sizes) + 1
            labels[y, x] = n
            stack = [(y, x)]
            min_y = max_y = y
            min_x = max_x = x
            first = (y, x)  # cells are seeded in scan order, so this is the first
            count = 0
            while stack:
                cy, cx = stack.pop()
                count += 1
                if cy < min_y:
                    min_y = cy
                elif cy > max_y:
                    max_y = cy
                if cx < min_x:
                    min_x = cx
                elif cx > max_x:
                    max_x = cx
                if cx > 0 and block[cy, cx - 1] and not labels[cy, cx - 1]:
                    labels[cy, cx - 1] = n
                    stack.append((cy, cx - 1))
                if cx + 1 < tw and block[cy, cx + 1] and not labels[cy, cx + 1]:
                    labels[cy, cx + 1] = n
                    stack.append((cy, cx + 1))
                if cy > 0 and block[cy - 1, cx] and not labels[cy - 1, cx]:
                    labels[cy - 1, cx] = n
                    stack.append((cy - 1, cx))
                if cy + 1 < th and block[cy + 1, cx] and not labels[cy + 1, cx]:
                    labels[cy + 1, cx] = n
                    stack.append((cy + 1, cx))
            bboxes.append((min_y, min_x, max_y, max_x))
            firsts.append(first)
            sizes.append(count)
    return labels, bboxes, firsts, sizes


def _components_tiled(grid: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    """Connected components over a cell grid, block by block.

    Labels are computed independently per block; neighbours are then compared
    along the shared borders (left/right and top/bottom rows) and unioned, so
    components spanning block borders become one component.
    """
    gh, gw = grid.shape
    dsu = _DSU()
    above: dict[int, np.ndarray] = {}
    for by in range(0, gh, _TILE_CELLS):
        bh = min(_TILE_CELLS, gh - by)
        left_ids: np.ndarray | None = None
        for bx in range(0, gw, _TILE_CELLS):
            bw = min(_TILE_CELLS, gw - bx)
            block = grid[by : by + bh, bx : bx + bw]
            if not block.any():
                left_ids = np.zeros(bh, dtype=np.int64)
                above[bx] = np.zeros(bw, dtype=np.int64)
                continue
            labels, bboxes, firsts, sizes = _label_block(block)
            # Node ids are offset by 1 so 0 keeps meaning "background".
            id_of = np.zeros(len(sizes) + 1, dtype=np.int64)
            for i, (bbox, first, size) in enumerate(zip(bboxes, firsts, sizes), start=1):
                y0, x0, y1, x1 = bbox
                id_of[i] = dsu.add((by + y0, bx + x0, by + y1, bx + x1), (by + first[0], bx + first[1]), size) + 1
            global_labels = id_of[labels]

            if left_ids is not None:  # merge with the block to the left
                for y in range(bh):
                    a = int(left_ids[y])
                    b = int(global_labels[y, 0])
                    if a and b:
                        dsu.union(a - 1, b - 1)
            prev_above = above.get(bx)  # merge with the block above
            if prev_above is not None:
                for x in range(bw):
                    a = int(prev_above[x])
                    b = int(global_labels[0, x])
                    if a and b:
                        dsu.union(a - 1, b - 1)

            left_ids = global_labels[:, -1].copy()
            above[bx] = global_labels[-1, :].copy()
    return dsu.components(_MIN_CELLS)


def _contained(a: Region, b: Region) -> float:
    """Fraction of region `a` covered by region `b`."""
    x0, y0 = max(a.x, b.x), max(a.y, b.y)
    x1, y1 = min(a.x + a.width, b.x + b.width), min(a.y + a.height, b.y + b.height)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return ((x1 - x0) * (y1 - y0)) / a.area


def detect_regions(image: Image.Image, max_regions: int = 8) -> list[Region]:
    """Detect key regions; returns boxes sorted by score, highest first."""
    # Only convert when necessary: convert() copies the full bitmap, which is
    # the single unavoidable full-image allocation here.
    img = image if image.mode == "RGB" else image.convert("RGB")
    w, h = img.size
    gh, gw = max(h // _CELL, 1), max(w // _CELL, 1)
    grids = _build_grids(img, gh, gw)

    regions: list[Region] = []
    for kind in ("red_alert", "amber_warning"):
        for gy0, gx0, gy1, gx1, size in _components_tiled(grids[kind]):
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
    for gy0, gx0, gy1, gx1, size in _components_tiled(grids["contrast_block"]):
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
