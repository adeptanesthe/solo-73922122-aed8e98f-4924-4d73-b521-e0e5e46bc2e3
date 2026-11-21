# Region-detection tiling: equivalence and performance

`detect_regions` (the CV stage behind every `/v1/diagnose*` call) used to hold
whole-image copies of three colour/luminance arrays at once:

| allocation | size at 7680×4320 (per array) |
|---|---|
| RGB array (`np.asarray`) | 99.5 MB |
| red mask, amber mask (bool) | 33.2 MB each |
| luminance `float32` | 132.7 MB |
| per-pixel intermediates in `_masks` (int16 r/g/b + comparisons) | ~50 MB |
| cell grids, BFS `seen` array | ~1 MB |

The new implementation only ever materialises one fixed 768×768 px tile
(`_TILE_CELLS = 128` cells × `_CELL = 6` px) at a time. Each tile is reduced
straight to the compact cell grid, then discarded. Connected components are
labelled per tile and merged across tile borders with a union-find, so no
full-image mask, luminance array or `seen` array is ever allocated.

Nothing about the detection thresholds (`r>140`, `g<130`, …), the 0.35 cell
coverage, the 55.0 contrast std, the `_MIN_CELLS=6` size floor, scoring,
containment suppression, sorting or the returned pixel boxes changed.

## Equivalence

`benchmarks/legacy_regions.py` is a frozen copy of the pre-change module; the
benchmark compares `(kind, x, y, width, height, score, notes)` tuples exactly.

```
$ python -m benchmarks.bench_regions --width 3840 --height 2160 --runs 3
ALL EQUAL: True
```

| image | regions | identical to legacy |
|---|---|---|
| `samples/fault_screenshot.png` (960×540) | 4 | yes |
| synthetic 3840×2160 | 8 | yes |
| synthetic 7680×4320 | 8 | yes |

`tests/test_regions.py` additionally checks:

* 24 random grids of sizes straddling tile boundaries (130×140, 129×257,
  256×300, 301×129, 1×1, 5×400) at densities 0.02–0.95 — component lists
  (bbox, size **and order**) match the legacy whole-image BFS;
* a cross-shaped component that crosses both tile borders is merged into one
  box with the exact union bounding box and cell count;
* full `detect_regions` output matches legacy on the committed sample and on a
  1900×1300 image with banners/dialogs deliberately placed on tile edges;
* changing `_TILE_CELLS` (16 / 64 / 128) does not change the result;
* a frozen golden (captured from the legacy implementation on a 2560×1440
  synthetic screenshot) is reproduced exactly by both implementations.

The API tests (`tests/test_api.py`) additionally confirm that `/docs` serves
Swagger and that `/v1/diagnose` (multipart) and `/v1/diagnose/json` return
identical evidence ids, coordinates, hypotheses and relations for the same
input.

Beyond the committed tests, randomized fuzzing (300 random grids of
1×1…300×300 cells at densities 0.05–0.9 with block sizes 1…1024 cells, and 44
random images of odd/unusual dimensions × tile sizes 8/17/64/128) found zero
differences. Region and evidence order is preserved because components are
ordered by their first cell in row-major scan order — the same discovery order
as the old BFS. That keeps evidence ids (`E1…En`), and therefore every
relation and hypothesis reference, stable.

## Performance

Measured on this 2-core Linux container, Python 3.11, numpy 2.4.6. Wall time is
the best of 3 runs (2 for 8K); memory is peak RSS and peak traced Python
allocation from a separate process, so tracing overhead does not pollute the
timings.

| image | legacy time | tiled time | speed-up | legacy peak RSS | tiled peak RSS | legacy Python peak | tiled Python peak |
|---|---|---|---|---|---|---|---|
| 960×540 | 0.98 s | 0.09 s | 10.9× | 50.3 MB | 47.8 MB | 10.0 MB | 8.5 MB |
| 3840×2160 | 15.67 s | 0.99 s | 15.8× | 280.3 MB | 88.5 MB | 158.6 MB | 14.8 MB |
| 7680×4320 | 65.65 s | 3.60 s | 18.2× | 953.4 MB | 186.0 MB | 633.8 MB | 16.8 MB |

At the HTTP level, `/v1/diagnose/json` on the 3840×2160 synthetic JPEG
(1.4 MB, heuristic backend) completes in ~1.2 s end to end, with the CV stage
now ~1 s of that (it was ~16 s).

The Python-level allocation peak is now flat (~17 MB at 33 Mpx), i.e. it is
bounded by the tile size rather than the image. The residual RSS above that is
the decoded source image (a 7680×4320 RGB bitmap is ~95 MB) plus the 768×768
tile and interpreter/libs; that decode already happened in the upload path
(`_decode_image`) before this change.

Tile-size sweep on 3840×2160 (peak RSS, same 8 regions):

| `_TILE_CELLS` | px per tile | peak RSS |
|---|---|---|
| 32 | 192 | 97 MB |
| 64 | 384 | 101 MB |
| 128 (default) | 768 | 117 MB |
| 256 | 1536 | 164 MB |
| 512 | 3072 | 261 MB |

128 cells keeps the working set small while amortising per-tile overhead.

## Reproducing

```bash
python -m benchmarks.bench_regions                       # 3840×2160, 3 runs
python -m benchmarks.bench_regions --width 7680 --height 4320 --runs 2
python -m pytest tests/test_regions.py -v                # equivalence tests
```

The benchmark writes its synthetic screenshot to
`benchmarks/_large_sample.png` (excluded from git); drop it with `rm`.
