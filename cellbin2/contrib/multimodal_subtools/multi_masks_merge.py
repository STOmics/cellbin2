#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Merge multiple 0/1 masks by priority.

The order of --masks defines priority:
    mask_1 > mask_2 > mask_3 > ...

A lower-priority mask is kept only where it does not conflict with the
already-merged higher-priority masks.

Output files:
    <output_dir>/merged_mask.tif
    <output_dir>/mask_01_base.tif
    <output_dir>/mask_02_kept.tif
    <output_dir>/mask_03_kept.tif
    ...

Example:
    python merge_multi_masks_priority.py \
        --masks membrane_mask.tif nuclei_mask.tif interior_mask.tif \
        -o ./merge_output \
        --overlap-threshold 0.2
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from skimage.measure import label

try:
    from cellbin2.image import cbimread, cbimwrite
except ImportError:
    cbimread = None
    cbimwrite = None


def to_binary(mask: np.ndarray) -> np.ndarray:
    """
    Convert mask to 0/1 uint8.
    """
    return (mask > 0).astype(np.uint8)


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


def check_same_shape(masks: List[np.ndarray], mask_paths: List[str]):
    """
    Ensure all masks have exactly the same shape.
    """
    base_shape = masks[0].shape

    for idx, mask in enumerate(masks[1:], start=2):
        if mask.shape != base_shape:
            raise ValueError(
                "Mask shape mismatch: "
                f"mask_01={base_shape}, "
                f"mask_{idx:02d}={mask.shape}, "
                f"path={mask_paths[idx - 1]}"
            )


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


def keep_large_fragments(
    original_mask: np.ndarray,
    filtered_mask: np.ndarray,
    threshold: float = 0.2,
) -> np.ndarray:
    """
    Keep only large remaining fragments of each original object.

    For each connected component in original_mask, a remaining fragment is kept
    only if:

        fragment_area / original_component_area >= threshold
    """
    original_mask = to_binary(original_mask)
    filtered_mask = to_binary(filtered_mask)

    result = np.zeros_like(filtered_mask, dtype=np.uint8)

    contours, _ = cv2.findContours(
        original_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        filtered_roi = filtered_mask[y:y + h, x:x + w]
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


def remove_added_mask_boundary(mask: np.ndarray) -> np.ndarray:
    """
    Remove 1-pixel contour boundary from a newly added low-priority mask.

    This avoids the added regions introducing boundary pixels that connect or
    interfere with neighboring cells.
    """
    mask = to_binary(mask)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    boundary = np.zeros_like(mask, dtype=np.uint8)
    cv2.drawContours(boundary, contours, -1, 1, 1)

    mask_without_boundary = np.where(
        boundary > 0,
        0,
        mask,
    ).astype(np.uint8)

    return mask_without_boundary


def merge_multi_masks_by_priority(
    masks: List[np.ndarray],
    overlap_threshold: float = 0.2,
    break_diagonal: bool = True,
    remove_boundary_for_added: bool = True,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Merge multiple 0/1 masks by priority.

    The first mask has the highest priority. Later masks only fill areas not
    occupied by earlier masks.

    Args:
        masks:
            List of masks sorted from high priority to low priority.
        overlap_threshold:
            Minimum remaining fragment ratio for every lower-priority mask.
        break_diagonal:
            Whether to break diagonal-only connections in final merged mask.
        remove_boundary_for_added:
            Whether to remove 1-pixel boundary from every lower-priority mask
            after conflict removal and fragment filtering.

    Returns:
        kept_masks:
            kept_masks[0] is the highest-priority base mask.
            kept_masks[1:] are the actually retained regions from lower-priority
            masks after conflict removal, fragment filtering, and optional
            boundary removal.
        merged_mask:
            Final merged 0/1 mask.
    """
    if len(masks) < 2:
        raise ValueError("At least two masks are required.")

    masks = [to_binary(mask) for mask in masks]

    base_shape = masks[0].shape
    for idx, mask in enumerate(masks[1:], start=2):
        if mask.shape != base_shape:
            raise ValueError(
                f"Mask shape mismatch: mask_01={base_shape}, "
                f"mask_{idx:02d}={mask.shape}"
            )

    merged_mask = masks[0].copy().astype(np.uint8)
    kept_masks = [merged_mask.copy()]

    for idx, current_mask in enumerate(masks[1:], start=2):
        # 1. Remove pixels already occupied by all higher-priority masks.
        current_without_higher_priority = np.where(
            merged_mask > 0,
            0,
            current_mask,
        ).astype(np.uint8)

        # 2. Keep only sufficiently large remaining fragments from this mask.
        current_kept = keep_large_fragments(
            original_mask=current_mask,
            filtered_mask=current_without_higher_priority,
            threshold=overlap_threshold,
        )

        # 3. Remove boundary only for newly added lower-priority regions.
        if remove_boundary_for_added:
            current_kept = remove_added_mask_boundary(current_kept)

        # 4. Add this retained mask into the accumulated result.
        merged_mask = cv2.bitwise_or(merged_mask, current_kept).astype(np.uint8)
        kept_masks.append(current_kept.astype(np.uint8))

        print(
            f"[Info] mask_{idx:02d}: "
            f"input_pixels={int(current_mask.sum())}, "
            f"kept_pixels={int(current_kept.sum())}"
        )

    # 5. Break diagonal-only connections in the final merged mask.
    if break_diagonal:
        merged_mask = break_diagonal_connections(merged_mask)

    return kept_masks, merged_mask.astype(np.uint8)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge multiple 0/1 masks by priority."
    )

    parser.add_argument(
        "--masks",
        nargs="+",
        required=True,
        help=(
            "Mask paths sorted by priority from high to low. "
            "Example: --masks mask_high.tif mask_mid.tif mask_low.tif"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output folder.",
    )
    parser.add_argument(
        "--overlap-threshold",
        type=float,
        default=0.2,
        help=(
            "Minimum remaining fragment ratio for lower-priority objects. "
            "Default: 0.2"
        ),
    )
    parser.add_argument(
        "--no-break-diagonal",
        action="store_true",
        help="Do not break diagonal-only connections in final merged mask.",
    )
    parser.add_argument(
        "--keep-added-boundary",
        action="store_true",
        help=(
            "Keep boundaries of lower-priority added regions. "
            "By default, 1-pixel boundaries are removed from added masks."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if len(args.masks) < 2:
        raise ValueError("Please provide at least two masks after --masks.")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    masks = [read_mask(path) for path in args.masks]
    check_same_shape(masks, args.masks)

    kept_masks, merged_mask = merge_multi_masks_by_priority(
        masks=masks,
        overlap_threshold=args.overlap_threshold,
        break_diagonal=not args.no_break_diagonal,
        remove_boundary_for_added=not args.keep_added_boundary,
    )

    merged_output_path = output_dir / "merged_mask.tif"
    write_mask(merged_output_path, merged_mask)

    # Save each priority layer after processing, useful for checking what each
    # mask actually contributed to the final merged result.
    for idx, kept_mask in enumerate(kept_masks, start=1):
        if idx == 1:
            output_path = output_dir / "mask_01_base.tif"
        else:
            output_path = output_dir / f"mask_{idx:02d}_kept.tif"
        write_mask(output_path, kept_mask)

    print(f"[Success] Saved merged mask: {merged_output_path}")
    print(f"[Success] Saved priority layers to: {output_dir}")


if __name__ == "__main__":
    main()
