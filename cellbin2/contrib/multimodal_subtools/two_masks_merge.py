#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Merge two 0/1 masks by priority.

Primary mask has higher priority.
Secondary mask is kept only where it does not conflict with primary mask.

Output files:
    <output_dir>/merged_mask.tif
    <output_dir>/secondary_kept.tif

Example:
    python merge_two_masks_priority.py \
        --primary membrane_mask.tif \
        --secondary nuclei_mask.tif \
        -o ./merge_output \
        --overlap-threshold 0.2
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from skimage.measure import label


try:
    from cellbin2.image import cbimread, cbimwrite
except ImportError:
    cbimread = None
    cbimwrite = None


def read_mask(path: str) -> np.ndarray:
    """
    Read 0/1 mask image.
    """
    path = str(path)

    if cbimread is not None:
        mask = cbimread(path, only_np=True)
    else:
        mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"Failed to read mask: {path}")
        if mask.ndim == 3:
            mask = mask[:, :, 0]

    return to_binary(mask)


def write_mask(path: str, mask: np.ndarray):
    """
    Write mask as 0/1 uint8 tif.
    """
    path = str(path)
    mask = to_binary(mask)

    if cbimwrite is not None:
        cbimwrite(path, mask)
    else:
        ok = cv2.imwrite(path, mask)
        if not ok:
            raise IOError(f"Failed to write mask: {path}")


def to_binary(mask: np.ndarray) -> np.ndarray:
    """
    Convert mask to 0/1 uint8.
    """
    return (mask > 0).astype(np.uint8)


def break_diagonal_connections(binary_mask: np.ndarray) -> np.ndarray:
    """
    Break diagonal-only connections in a binary mask.
    """
    m = to_binary(binary_mask)
    out = m.copy()

    a = m[:-1, :-1]
    b = m[:-1, 1:]
    c = m[1:, :-1]
    d = m[1:, 1:]

    # Pattern:
    # 1 0
    # 0 1
    pattern1 = (a == 1) & (b == 0) & (c == 0) & (d == 1)

    # Pattern:
    # 0 1
    # 1 0
    pattern2 = (a == 0) & (b == 1) & (c == 1) & (d == 0)

    out[1:, 1:][pattern1] = 0
    out[1:, :-1][pattern2] = 0

    return out.astype(np.uint8)


def keep_large_secondary_fragments(
    original_secondary_mask: np.ndarray,
    filtered_secondary_mask: np.ndarray,
    threshold: float = 0.2,
) -> np.ndarray:
    """
    Keep only large remaining fragments of each original secondary object.

    For each connected component in original_secondary_mask:
        keep a remaining fragment only if:

            fragment_area / original_component_area >= threshold
    """
    original_secondary_mask = to_binary(original_secondary_mask)
    filtered_secondary_mask = to_binary(filtered_secondary_mask)

    result = np.zeros_like(filtered_secondary_mask, dtype=np.uint8)

    contours, _ = cv2.findContours(
        original_secondary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        filtered_roi = filtered_secondary_mask[y:y + h, x:x + w]
        contour_roi = contour - np.array([x, y])

        original_roi_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(original_roi_mask, [contour_roi], color=1)

        original_area = int(original_roi_mask.sum())
        if original_area == 0:
            continue

        filtered_inside_original = cv2.bitwise_and(
            filtered_roi,
            original_roi_mask,
        )

        labeled_fragments = label(filtered_inside_original, connectivity=1)
        fragment_ids = np.unique(labeled_fragments)
        fragment_ids = fragment_ids[fragment_ids != 0]

        kept_roi = np.zeros((h, w), dtype=np.uint8)

        for frag_id in fragment_ids:
            frag_mask = labeled_fragments == frag_id
            frag_area = int(frag_mask.sum())
            frag_ratio = frag_area / original_area

            if frag_ratio >= threshold:
                kept_roi[frag_mask] = 1

        result[y:y + h, x:x + w] = np.maximum(
            result[y:y + h, x:x + w],
            kept_roi,
        )

    return result.astype(np.uint8)


def remove_secondary_boundary(secondary_mask: np.ndarray) -> np.ndarray:
    """
    Remove 1-pixel contour boundary from secondary mask.

    This is important for downstream cell matrix extraction because the
    secondary-added regions should not introduce boundary pixels that connect
    or interfere with neighboring cell masks.
    """
    secondary_mask = to_binary(secondary_mask)

    contours, _ = cv2.findContours(
        secondary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    secondary_boundary = np.zeros_like(secondary_mask, dtype=np.uint8)
    cv2.drawContours(secondary_boundary, contours, -1, 1, 1)

    secondary_without_boundary = np.where(
        secondary_boundary > 0,
        0,
        secondary_mask,
    ).astype(np.uint8)

    return secondary_without_boundary


def merge_two_masks_by_priority(
    primary_mask: np.ndarray,
    secondary_mask: np.ndarray,
    overlap_threshold: float = 0.2,
    break_diagonal: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Merge two 0/1 masks by priority.

    Primary mask has higher priority.
    Secondary mask only fills regions not occupied by primary.

    Returns:
        secondary_kept:
            Secondary mask after conflict removal, fragment filtering,
            and boundary removal.

        merged_mask:
            Final merged 0/1 mask.
    """
    primary = to_binary(primary_mask)
    secondary = to_binary(secondary_mask)

    if primary.shape != secondary.shape:
        raise ValueError(
            f"Mask shape mismatch: primary={primary.shape}, "
            f"secondary={secondary.shape}"
        )

    # 1. Remove secondary pixels covered by primary.
    secondary_without_primary = np.where(
        primary > 0,
        0,
        secondary,
    ).astype(np.uint8)

    # 2. Keep only sufficiently large secondary fragments.
    secondary_kept = keep_large_secondary_fragments(
        original_secondary_mask=secondary,
        filtered_secondary_mask=secondary_without_primary,
        threshold=overlap_threshold,
    )

    # 3. Always remove secondary boundary.
    secondary_kept = remove_secondary_boundary(secondary_kept)

    # 4. Merge primary and kept secondary.
    merged_mask = cv2.bitwise_or(primary, secondary_kept).astype(np.uint8)

    # 5. Break diagonal-only connections.
    if break_diagonal:
        merged_mask = break_diagonal_connections(merged_mask)

    return secondary_kept, merged_mask


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge two 0/1 masks by priority."
    )

    parser.add_argument(
        "--primary",
        required=True,
        help="Path to high-priority 0/1 mask.",
    )
    parser.add_argument(
        "--secondary",
        required=True,
        help="Path to low-priority 0/1 mask.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help=(
            "Output folder. The script will save merged_mask.tif "
            "and secondary_kept.tif inside this folder."
        ),
    )
    parser.add_argument(
        "--overlap-threshold",
        type=float,
        default=0.2,
        help=(
            "Minimum remaining fragment ratio for secondary objects. "
            "Default: 0.2"
        ),
    )
    parser.add_argument(
        "--no-break-diagonal",
        action="store_true",
        help="Do not break diagonal-only connections in final merged mask.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    primary_mask = read_mask(args.primary)
    secondary_mask = read_mask(args.secondary)

    secondary_kept, merged_mask = merge_two_masks_by_priority(
        primary_mask=primary_mask,
        secondary_mask=secondary_mask,
        overlap_threshold=args.overlap_threshold,
        break_diagonal=not args.no_break_diagonal,
    )

    merged_output_path = output_dir / "merged_mask.tif"
    secondary_output_path = output_dir / "secondary_kept.tif"

    write_mask(merged_output_path, merged_mask)
    write_mask(secondary_output_path, secondary_kept)

    print(f"[Success] Saved merged mask: {merged_output_path}")
    print(f"[Success] Saved kept secondary mask: {secondary_output_path}")


if __name__ == "__main__":
    main()