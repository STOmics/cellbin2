"""
cell_id_sort.py
=========================
Mask-only cell-ID assignment that replicates the geftools snake-scan +
fixed-seed shuffle algorithm (src/cgefCellgem.cpp + src/utils.cpp).

All cells in the mask are assumed to have expression — no bgef / DNB filter —
so the result is a *dense* 1..N mapping.

Public API
----------
  build_cell_id_map_from_mask(mask, apply_contour_filter=True, connectivity=8)
      → (dict[cc_label, cell_id], cc_label_map)

  apply_cell_ids_to_mask(cc_label_map, id_map)
      → np.ndarray (uint32, pixel = cell_id)

Algorithm (rebuildCellIdMapForSubset in cgefCellgem.cpp)
---------
  1  Binary mask → cv2.connectedComponentsWithStats → CC labels + centroids.
  2  Optional contour filter: keep labels whose bounding-rect matches a contour
     with > 3 vertices (mirrors C++ getCell / buildContourFilteredLabels).
  3  Sort surviving cells by (y_int, label) ascending — total order.
  4  Snake scan: even rows sort x ascending, odd rows sort x descending;
     both use label as secondary tiebreak.
  5  Fixed-seed (2193796729) Fisher-Yates shuffle of the snake-scan order.
  6  cell_id = position_in_shuffled_order + 1  (1-based, dense 1..N).

References
----------
  geftools/src/cgefCellgem.cpp  – rebuildCellIdMapForSubset(),
                                   buildContourFilteredLabels(), getCell()
  geftools/src/utils.cpp        – shuffleCellOrder()
  geftools/src/utils.h          – kCellIdShuffleSeed = 2193796729
"""

from __future__ import annotations

import itertools
from typing import NamedTuple

import cv2
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

_CELL_ID_SHUFFLE_SEED: int = 2193796729
_MT_RANGE: int = 1 << 32   # std::mt19937 range = 2^32


# ──────────────────────────────────────────────────────────────────────────────
# MT19937 — portable implementation that matches std::mt19937 (GCC / MSVC)
# ──────────────────────────────────────────────────────────────────────────────

class _MT19937:
    """Minimal Mersenne Twister 19937, output-identical to std::mt19937."""

    _N          = 624
    _M          = 397
    _MATRIX_A   = 0x9908_B0DF
    _UPPER_MASK = 0x8000_0000
    _LOWER_MASK = 0x7FFF_FFFF

    def __init__(self, seed: int) -> None:
        mt    = [0] * self._N
        mt[0] = seed & 0xFFFF_FFFF
        for i in range(1, self._N):
            mt[i] = (1812433253 * (mt[i - 1] ^ (mt[i - 1] >> 30)) + i) & 0xFFFF_FFFF
        self._mt    = mt
        self._index = self._N

    def _generate(self) -> None:
        mt    = self._mt
        mag01 = (0, self._MATRIX_A)
        for i in range(self._N):
            y     = (mt[i] & self._UPPER_MASK) | (mt[(i + 1) % self._N] & self._LOWER_MASK)
            mt[i] = mt[(i + self._M) % self._N] ^ (y >> 1) ^ mag01[y & 1]
        self._index = 0

    def next_u32(self) -> int:
        if self._index >= self._N:
            self._generate()
        y = self._mt[self._index]
        self._index += 1
        y ^= y >> 11
        y ^= (y << 7)  & 0x9D2C_5680
        y ^= (y << 15) & 0xEFC6_0000
        y ^= y >> 18
        # mt[] entries stay 32-bit; tempering preserves width — no final mask needed.
        return y

    def uniform(self, n: int) -> int:
        """Uniform integer in [0, n) via rejection sampling — mirrors utils.cpp."""
        # n must fit in uint32_t: [1, 2^32-1].  n=2^32 would make limit=0 and
        # loop forever (C++ receives n truncated to uint32_t(0) = division-by-zero).
        assert 1 <= n < _MT_RANGE, f"uniform: n={n} outside [1, {_MT_RANGE - 1}]"
        limit = _MT_RANGE - (_MT_RANGE % n)
        while True:
            r = self.next_u32()
            if r < limit:
                return r % n


# ──────────────────────────────────────────────────────────────────────────────
# Internal centroid → cell_id pipeline
# ──────────────────────────────────────────────────────────────────────────────

class _Cell(NamedTuple):
    y: int      # truncated centroid row     (primary sort key)
    x: int      # truncated centroid column  (snake-scan sort key)
    label: int  # CC label (1-based)


def _snake_order(cells: list[_Cell]) -> list[int]:
    """
    Phase 2–3: sort by (y, label) → snake scan per row → ordered CC labels.

    NOTE: C++ uses unstable sort by y only (non-deterministic for equal-y).
    The label tiebreak here makes Python output reproducible.  The per-row
    x-sort in the snake scan makes the y-tiebreak non-load-bearing for
    final cell_id values.

    Does NOT mutate the caller's list (sorts a copy).
    """
    ordered = sorted(cells, key=lambda c: (c.y, c.label))
    snake: list[int] = []
    for row_idx, (_, row_iter) in enumerate(itertools.groupby(ordered, key=lambda c: c.y)):
        row = sorted(row_iter, key=lambda c: (c.x if row_idx % 2 == 0 else -c.x, c.label))
        snake.extend(c.label for c in row)
    return snake


def _shuffle_inplace(order: list[int]) -> None:
    """Fixed-seed Fisher-Yates shuffle — identical to utils.cpp::shuffleCellOrder."""
    mt = _MT19937(_CELL_ID_SHUFFLE_SEED)
    for k in range(len(order), 1, -1):
        j = mt.uniform(k)
        order[k - 1], order[j] = order[j], order[k - 1]


def _assign_ids(cells: list[_Cell]) -> dict[int, int]:
    """Snake-scan + shuffle → {cc_label: cell_id}."""
    if not cells:
        return {}
    snake = _snake_order(cells)
    _shuffle_inplace(snake)
    return {label: k + 1 for k, label in enumerate(snake)}


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def _contour_valid_rects(binary: np.ndarray) -> set[tuple[int, int, int, int]]:
    """Return bounding-rects of external contours with > 3 vertices.

    Note: two CCs that share an identical bounding rect both inherit the same
    pass/fail decision (inherited from the C++ rect_to_cidx map behaviour).
    """
    # [-2] selects contours from both OpenCV 3.x (image, contours, hierarchy)
    # and 4.x (contours, hierarchy) return tuples.
    contours = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    return {tuple(map(int, cv2.boundingRect(c))) for c in contours if len(c) > 3}


def build_cell_id_map_from_mask(
    mask: np.ndarray,
    apply_contour_filter: bool = True,
    connectivity: int = 8,
) -> tuple[dict[int, int], np.ndarray]:
    """
    Assign snake-scan + shuffle cell IDs to every cell in a mask.

    All cells are assumed to have expression, so the result is a dense
    1..N mapping (no gaps).

    Parameters
    ----------
    mask : np.ndarray, 2-D
        Any non-zero pixel is foreground.  Accepts binary (0/1 or 0/255) or
        pre-labeled (each cell has its own non-zero value).  Internally
        binarised before ``cv2.connectedComponentsWithStats``.
    apply_contour_filter : bool
        True (default) — keep only CC labels whose bounding-rect matches a
        contour with > 3 vertices, mirroring the C++ filter in
        ``getCell()`` / ``buildContourFilteredLabels()``.
        False — keep all non-zero connected components.
    connectivity : int
        4 or 8 (default 8) for ``cv2.connectedComponentsWithStats``.

    Returns
    -------
    id_map : dict[int, int]
        ``{cc_label (1-based) → cell_id (1-based, dense 1..N)}``.
        Empty dict when no cells are found.
    cc_label_map : np.ndarray (int32)
        Same shape as *mask*; pixel value = CC label (0 = background).
        Pass to :func:`apply_cell_ids_to_mask` to build the relabeled output.

    Raises
    ------
    ValueError   if *mask* is not 2-D.
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {mask.shape}")

    if apply_contour_filter and connectivity != 8:
        raise ValueError(
            "apply_contour_filter=True requires connectivity=8: "
            "cv2.findContours always uses 8-connected boundary tracing, "
            "which is inconsistent with connectivity=4 CC labeling "
            "(matching the C++ rebuildCellIdMapForSubset which hardcodes 8-connectivity)."
        )

    binary = (mask > 0).astype(np.uint8)
    retval, cc_label_map, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=connectivity, ltype=cv2.CV_32S
    )

    if retval <= 1:
        return {}, cc_label_map

    if apply_contour_filter:
        valid_rects = _contour_valid_rects(binary)
        labels = [
            lbl for lbl in range(1, retval)
            if (
                int(stats[lbl, cv2.CC_STAT_LEFT]),
                int(stats[lbl, cv2.CC_STAT_TOP]),
                int(stats[lbl, cv2.CC_STAT_WIDTH]),
                int(stats[lbl, cv2.CC_STAT_HEIGHT]),
            ) in valid_rects
        ]
    else:
        labels = list(range(1, retval))

    if not labels:
        return {}, cc_label_map

    cells = [
        _Cell(y=int(centroids[lbl, 1]), x=int(centroids[lbl, 0]), label=lbl)
        for lbl in labels
    ]
    return _assign_ids(cells), cc_label_map


def apply_cell_ids_to_mask(
    cc_label_map: np.ndarray,
    id_map: dict[int, int],
) -> np.ndarray:
    """
    Convert a CC label map to a cell_id label map.

    Parameters
    ----------
    cc_label_map : np.ndarray
        Second return value of :func:`build_cell_id_map_from_mask`.
    id_map : dict[int, int]
        First return value of :func:`build_cell_id_map_from_mask`.

    Returns
    -------
    np.ndarray (uint32)
        New mask, same shape as *cc_label_map*.
        Pixel value = new cell_id; background and filtered cells = 0.
    """
    if cc_label_map.size == 0:
        return np.zeros(cc_label_map.shape, dtype=np.uint32)

    max_cc = int(cc_label_map.max())
    lut    = np.zeros(max_cc + 1, dtype=np.uint32)
    for cc_lbl, cell_id in id_map.items():
        if 0 <= cc_lbl <= max_cc:
            lut[cc_lbl] = cell_id
    return lut[cc_label_map]


# ──────────────────────────────────────────────────────────────────────────────
# Self-test  (python cell_id_sort.py)
# ──────────────────────────────────────────────────────────────────────────────

def _self_test() -> None:
    # ── MT19937 tempering check ───────────────────────────────────────────────
    # C++ standard (N4140 §26.5.3.1): std::mt19937(5489).first = 3499211612.
    mt = _MT19937(5489)
    v  = mt.next_u32()
    assert v == 3499211612, (
        f"MT19937 tempering mismatch: got {v}, expected 3499211612. "
        "Check constants in _MT19937.next_u32()."
    )
    print(f"[PASS] MT19937 reference (seed=5489 → {v})")

    # ── shuffle golden value ──────────────────────────────────────────────────
    # Verified against C++ std::mt19937(2193796729) + Fisher-Yates from k=5.
    s = [10, 20, 30, 40, 50]
    _shuffle_inplace(s)
    assert s == [30, 20, 10, 40, 50], f"shuffle golden mismatch: {s}"
    print(f"[PASS] shuffle golden N=5: [10,20,30,40,50] → {s}")

    # ── mask tests ────────────────────────────────────────────────────────────
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[1:6,   1:6]   = 1   # cell A (top-left)
    mask[12:17, 12:17] = 1   # cell B (bottom-right)
    id_map, cc_lbl = build_cell_id_map_from_mask(mask)
    # CC scan order (top→bottom, left→right): label 1 = A, label 2 = B.
    # Snake: row 0 (y≈3) ascending → [1]; row 1 (y≈14) descending → [2].
    # Shuffle N=2, seed 2193796729: first uniform(2) returns 1 → no swap.
    assert id_map == {1: 1, 2: 2}, f"expected {{1:1, 2:2}}, got {id_map}"
    print(f"[PASS] two-cell mask: id_map = {id_map}")

    out   = apply_cell_ids_to_mask(cc_lbl, id_map)
    ids_a = set(out[1:6,   1:6].flat)
    ids_b = set(out[12:17, 12:17].flat)
    assert int(out[0, 0]) == 0, "background must be 0"
    assert ids_a == {np.uint32(1)} and ids_b == {np.uint32(2)}, \
        f"expected A={{1}}, B={{2}}: got A={ids_a} B={ids_b}"
    print(f"[PASS] apply_cell_ids_to_mask: A={ids_a} B={ids_b}, background=0")

    id_map2, cc2 = build_cell_id_map_from_mask(np.zeros((10, 10), dtype=np.uint8))
    assert id_map2 == {} and (cc2 == 0).all(), "empty mask should give empty id_map"
    print("[PASS] empty mask → empty id_map")

    print("\nAll tests passed.")


if __name__ == "__main__":
    _self_test()
