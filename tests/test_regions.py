"""Region-detection tests: tiled implementation must match the legacy one."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from app.vision import regions as regions_mod
from app.vision.regions import _CELL, _TILE_CELLS, _components_tiled, detect_regions
from benchmarks.legacy_regions import _components as legacy_components
from benchmarks.legacy_regions import detect_regions as legacy_detect_regions


def _as_tuple(regions):
    return [(r.kind, r.x, r.y, r.width, r.height, r.score, tuple(r.notes)) for r in regions]


def _paint(img: Image.Image, w: int, h: int) -> Image.Image:
    """Error colours + contrast blocks that straddle tile boundaries."""
    d = ImageDraw.Draw(img)
    tile_px = _TILE_CELLS * _CELL
    # horizontal red band crossing many tiles vertically
    d.rectangle([0, h // 10, w, h // 10 + 90], fill=(198, 40, 40))
    # amber strip starting just left of the first tile boundary
    d.rectangle([tile_px - 30, h // 4, w, h // 4 + 36], fill=(255, 171, 0))
    # dialog straddling the boundary both horizontally and vertically
    d.rectangle(
        [tile_px - 60, tile_px - 60, tile_px + 220, tile_px + 140],
        fill=(255, 255, 255),
        outline=(60, 60, 60),
        width=3,
    )
    d.rectangle([tile_px - 60, tile_px - 60, tile_px + 220, tile_px - 30], fill=(58, 58, 60))
    return img


@pytest.mark.parametrize("shape", [(130, 140), (129, 257), (256, 300), (301, 129), (1, 1), (5, 400)])
@pytest.mark.parametrize("density", [0.02, 0.35, 0.6, 0.95])
def test_tiled_components_match_legacy_on_random_grids(shape, density):
    gh, gw = shape
    rng = np.random.default_rng(gh * 1000 + gw + int(density * 100))
    grid = rng.random((gh, gw)) < density
    assert _components_tiled(grid) == legacy_components(grid)


def test_tiled_components_merge_across_tile_boundaries():
    gh, gw = 300, 300
    grid = np.zeros((gh, gw), dtype=bool)
    grid[40:260, 120:135] = True  # vertical band crossing the x=128 tile edge
    grid[120:135, 10:290] = True  # horizontal band crossing the y=128 tile edge
    parsed = _components_tiled(grid)
    assert parsed == legacy_components(grid)
    assert len(parsed) == 1
    (y0, x0, y1, x1, size) = parsed[0]
    assert (y0, x0, y1, x1) == (40, 10, 260, 290)
    assert size == int(grid.sum())


def test_detect_regions_tiled_matches_legacy_on_large_image():
    w, h = 1900, 1300
    img = _paint(Image.new("RGB", (w, h), (245, 246, 248)), w, h)
    assert _as_tuple(detect_regions(img, max_regions=8)) == _as_tuple(
        legacy_detect_regions(img, max_regions=8)
    )
    assert detect_regions(img)


def test_detect_regions_matches_legacy_on_sample_image():
    from pathlib import Path

    sample = Path(__file__).resolve().parent.parent / "samples" / "fault_screenshot.png"
    if not sample.exists():
        pytest.skip("sample screenshot not generated")
    with Image.open(sample) as img:
        img.load()
        assert _as_tuple(detect_regions(img)) == _as_tuple(legacy_detect_regions(img))


def test_tile_size_does_not_change_result():
    w, h = 1500, 900
    img = _paint(Image.new("RGB", (w, h), (245, 246, 248)), w, h)
    original = regions_mod._TILE_CELLS
    try:
        results = []
        for tile in (16, 64, _TILE_CELLS):
            regions_mod._TILE_CELLS = tile
            results.append(_as_tuple(detect_regions(img, max_regions=8)))
    finally:
        regions_mod._TILE_CELLS = original
    assert results[0] == results[1] == results[2]


def test_large_synthetic_image_matches_frozen_golden():
    """Golden captured from the legacy implementation for a 2560x1440 image."""
    from benchmarks.bench_regions import _synthetic

    golden = [
        ("red_alert", 0, 132, 2556, 132, 0.98, ("red_alert covers 8.3% of image",)),
        ("amber_warning", 0, 360, 2556, 66, 0.706, ("amber_warning covers 2.9% of image",)),
        ("red_alert", 756, 864, 42, 42, 0.553, ("red_alert covers 0.0% of image",)),
        ("red_alert", 1608, 54, 30, 42, 0.552, ("red_alert covers 0.0% of image",)),
        ("red_alert", 1014, 642, 36, 42, 0.552, ("red_alert covers 0.0% of image",)),
        ("red_alert", 1056, 768, 30, 36, 0.552, ("red_alert covers 0.0% of image",)),
        ("red_alert", 1416, 828, 36, 36, 0.552, ("red_alert covers 0.0% of image",)),
        ("red_alert", 894, 984, 36, 42, 0.552, ("red_alert covers 0.0% of image",)),
    ]
    img = _synthetic(2560, 1440, seed=99)
    assert _as_tuple(detect_regions(img, max_regions=8)) == golden
    assert _as_tuple(legacy_detect_regions(img, max_regions=8)) == golden
