#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Void_segmentation
Function:
1. Extract raw void regions from cell mask
2. Extract boundary cells, cells inside raw void, and remaining cells
3. Merge raw void with boundary cells, then smooth and fill holes to generate final void
"""
"""
Usage example:
python Void_segmentation.py \
    --cell input_cell_mask.tif \
    --raw_void_out raw_void.tif \
    --void_out final_void.tif \
    --boundary_out boundary_cells.tif \
    --in_void_out cells_in_void.tif \
    --other_out cells_other.tif \
    --dilate 2 \
    --close 2 \
    --min_area 3000 \
    --threshold 5 \
    --smooth 3

Arguments:
--cell: Input cell mask image
--raw_void_out: Output path for raw void mask
--void_out: Output path for final void mask (merged with boundary cells)
--boundary_out: Output path for boundary cells mask
--in_void_out: Output path for cells completely inside void (excluding boundary cells)
--other_out: Output path for remaining cells (non-boundary and non-void)
--dilate: Dilation kernel size (default: 2)
--close: Morphological closing kernel size for tissue mask (default: 2)
--min_area: Minimum area threshold for filtering small void regions (default: 3000)
--threshold: Distance threshold for identifying boundary cells (default: 5)
--smooth: Smoothing kernel size for final void mask (default: 3)
"""
import cv2
import numpy as np
import tifffile
import argparse
from tqdm import tqdm

class CellBoundaryVoidExtractor:
    def __init__(self, dilate_size=2, close_size=2, min_void_area=3000,
                 distance_threshold=5, smooth_size=3, progress_bar=True):
        self.dilate_size = dilate_size
        self.close_size = close_size
        self.min_void_area = min_void_area
        self.distance_threshold = distance_threshold
        self.smooth_size = smooth_size
        self.progress_bar = progress_bar

    # ==== Original methods unchanged ====
    def get_void_mask_from_cell(self, cell_mask: np.ndarray) -> np.ndarray:
        cell_bin = (cell_mask > 0).astype(np.uint8)
        if self.dilate_size > 0:
            kernel = np.ones((self.dilate_size, self.dilate_size), np.uint8)
            cell_bin = cv2.dilate(cell_bin, kernel, iterations=1)
        contours, _ = cv2.findContours(cell_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        tissue_mask = np.zeros_like(cell_bin)
        cv2.drawContours(tissue_mask, contours, -1, 1, thickness=cv2.FILLED)
        if self.close_size > 0:
            kernel = np.ones((self.close_size, self.close_size), np.uint8)
            tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, kernel)
        void_mask = cv2.bitwise_and((cell_bin == 0).astype(np.uint8), tissue_mask)
        return void_mask.astype(np.uint8)

    def remove_small_void(self, void_mask: np.ndarray) -> np.ndarray:
        mask = (void_mask > 0).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        cleaned_mask = np.zeros_like(mask)
        iterator = range(1, num_labels)
        if self.progress_bar:
            iterator = tqdm(iterator, desc="Filtering small void regions")
        for i in iterator:
            if stats[i, cv2.CC_STAT_AREA] >= self.min_void_area:
                cleaned_mask[labels == i] = 1
        return cleaned_mask

    def fill_holes(self, mask: np.ndarray) -> np.ndarray:
        mask_bin = (mask > 0).astype(np.uint8)
        inv_mask = 1 - mask_bin
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv_mask, connectivity=8)
        bg_label = labels[0, 0]
        filled_mask = mask_bin.copy()
        for i in range(1, num_labels):
            if i == bg_label:
                continue
            filled_mask[labels == i] = 1
        return filled_mask

    def extract_boundary_and_void_cells(self, cell_mask: np.ndarray, void_mask: np.ndarray) -> tuple:
        cell_bin = (cell_mask > 0).astype(np.uint8)
        void_bin = (void_mask > 0).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cell_bin, connectivity=8)
        max_label = labels.max()

        # fully inside void
        void_labels = labels[void_bin > 0]
        unique_void_labels, void_counts = np.unique(void_labels, return_counts=True)
        _, counts = np.unique(labels, return_counts=True)
        total_counts = np.zeros(max_label + 1, dtype=np.int32)
        total_counts[:len(counts)] = counts
        fully_inside_void = np.zeros(max_label + 1, dtype=bool)
        fully_inside_void[unique_void_labels] = (void_counts == total_counts[unique_void_labels])
        cells_in_void = fully_inside_void[labels].astype(np.uint8)

        # boundary cells
        dist_map = cv2.distanceTransform(1 - void_bin, cv2.DIST_L2, 5)
        flat_labels = labels.ravel()
        flat_dist = dist_map.ravel()
        min_dist_per_label = np.full(max_label + 1, np.inf, dtype=np.float32)
        np.minimum.at(min_dist_per_label, flat_labels, flat_dist)
        near_mask = (min_dist_per_label < self.distance_threshold)
        near_mask[0] = False
        near_mask &= ~fully_inside_void
        boundary_cells = near_mask[labels].astype(np.uint8)

        return boundary_cells, cells_in_void

    def extract_cells_other(self, cell_mask: np.ndarray, boundary_cells: np.ndarray, cells_in_void: np.ndarray) -> np.ndarray:
        cell_bin = (cell_mask > 0).astype(np.uint8)
        cells_other = cell_bin.copy()
        cells_other[(boundary_cells > 0) | (cells_in_void > 0)] = 0
        return cells_other.astype(np.uint8)

    # ==== New: Merge raw void with boundary cells to generate final void ====
    def merge_void_with_boundary(self, raw_void: np.ndarray, boundary_cells: np.ndarray) -> np.ndarray:
        merged = np.maximum(raw_void, boundary_cells)
        kernel = np.ones((self.smooth_size, self.smooth_size), np.uint8)
        merged = cv2.dilate(merged, kernel, iterations=1)
        merged = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, kernel)

        # Fill holes
        inv_mask = 1 - merged
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv_mask, connectivity=8)
        bg_label = labels[0, 0]
        filled = merged.copy()
        for i in range(1, num_labels):
            if i == bg_label:
                continue
            filled[labels == i] = 1
        return filled.astype(np.uint8)

    def process(self, cell_mask: np.ndarray) -> tuple:
        if cell_mask.ndim == 3:
            cell_mask = cv2.cvtColor(cell_mask, cv2.COLOR_RGB2GRAY)
        cell_mask = (cell_mask > 0).astype(np.uint8)

        print("Step 1/6: Calculating raw void regions...")
        raw_void = self.get_void_mask_from_cell(cell_mask)
        print("Step 2/6: Filtering small void regions...")
        raw_void = self.remove_small_void(raw_void)
        print("Step 3/6: Filling small holes in raw void...")
        raw_void = self.fill_holes(raw_void)
        print("Step 4/6: Extracting boundary cells and cells in void...")
        boundary_cells, cells_in_void = self.extract_boundary_and_void_cells(cell_mask, raw_void)
        print("Step 5/6: Extracting remaining cells...")
        cells_other = self.extract_cells_other(cell_mask, boundary_cells, cells_in_void)
        print("Step 6/6: Merging raw void with boundary cells to generate final void...")
        final_void = self.merge_void_with_boundary(raw_void, boundary_cells)

        print("Processing complete!")
        return raw_void, final_void, boundary_cells, cells_in_void, cells_other


def main():
    parser = argparse.ArgumentParser(description="Extract cell boundaries and void regions (raw_void + final void)")
    parser.add_argument("--cell", required=True, help="Input cell mask path")
    parser.add_argument("--raw_void_out", required=True, help="Output path for raw void mask")
    parser.add_argument("--void_out", required=True, help="Output path for final void mask")
    parser.add_argument("--boundary_out", required=True, help="Output path for boundary cells mask")
    parser.add_argument("--in_void_out", required=True, help="Output path for cells in void mask")
    parser.add_argument("--other_out", required=True, help="Output path for remaining cells mask")
    parser.add_argument("--dilate", type=int, default=2)
    parser.add_argument("--close", type=int, default=2)
    parser.add_argument("--min_area", type=int, default=3000)
    parser.add_argument("--threshold", type=int, default=5)
    parser.add_argument("--smooth", type=int, default=3)
    args = parser.parse_args()

    # Automatically create output directories
    import os
    for path in [args.raw_void_out, args.void_out, args.boundary_out, args.in_void_out, args.other_out]:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    cell_mask = tifffile.imread(args.cell)
    extractor = CellBoundaryVoidExtractor(
        dilate_size=args.dilate,
        close_size=args.close,
        min_void_area=args.min_area,
        distance_threshold=args.threshold,
        smooth_size=args.smooth
    )

    raw_void, final_void, boundary_cells, cells_in_void, cells_other = extractor.process(cell_mask)

    # Save results
    tifffile.imwrite(args.raw_void_out, (raw_void * 255).astype(np.uint8), compression="zlib")
    tifffile.imwrite(args.void_out, (final_void * 255).astype(np.uint8), compression="zlib")
    tifffile.imwrite(args.boundary_out, (boundary_cells * 255).astype(np.uint8), compression="zlib")
    tifffile.imwrite(args.in_void_out, (cells_in_void * 255).astype(np.uint8), compression="zlib")
    tifffile.imwrite(args.other_out, (cells_other * 255).astype(np.uint8), compression="zlib")

    print(f"✅ raw void: {args.raw_void_out}")
    print(f"✅ final void: {args.void_out}")
    print(f"✅ boundary cells: {args.boundary_out}")
    print(f"✅ cells in void: {args.in_void_out}")
    print(f"✅ other cells: {args.other_out}")



if __name__ == "__main__":
    main()
