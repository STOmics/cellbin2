#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Export cell mask + nuclear mask to GeoJSON with sorted cell_id.

Compatible with Python 3.8.

Features
--------
1. final_cell_mask is relabeled by cgef/cellgem-like block sort logic:
   - block id ascending
   - inside each block: centroid x descending, then centroid y descending
   - cell_id starts from 0
2. final_nuclear_mask is assigned to cell_id by nucleus centroid falling inside cell mask.
   Multiple nuclei can share the same cell_id.
3. Compatible with multimodal and non-multimodal modes:
   - If --save-path is omitted or contains no middle masks, all cell source = nuclear_expand.
   - If middle masks exist, infer source from interior/boundary/output_nuclei masks.
4. Optional relabeled TIFF output:
   - cell_id_plus1.tif: background=0, cell pixels=cell_id+1
   - nucleus_cell_id_plus1.tif: background=0, nucleus pixels=cell_id+1, unassigned nucleus=0

Example
-------
python cell_nuclear_geojson_py38.py \
  --cell-mask final_cell_mask.tif \
  --nuclear-mask final_nuclear_mask.tif \
  --out /path/to/output_dir \
  --no-relabel-tif
"""

from __future__ import print_function

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
from scipy import ndimage
from skimage.measure import regionprops
from skimage.morphology import remove_small_objects
from rasterio.features import shapes
import rasterio.transform


# -----------------------------------------------------------------------------
# Image IO
# -----------------------------------------------------------------------------

def read_tif(path):
    """Read TIFF. Prefer cellbin cbimread, fallback to tifffile."""
    path = str(path)
    try:
        from cellbin2.image import cbimread
        return cbimread(path, only_np=True)
    except Exception:
        try:
            from cellbin.image import cbimread
            return cbimread(path, only_np=True)
        except Exception:
            import tifffile
            return tifffile.imread(path)


def write_tif(path, arr):
    """Write TIFF. Prefer cellbin cbimwrite, fallback to tifffile."""
    path = str(path)
    try:
        from cellbin2.image import cbimwrite
        cbimwrite(path, arr)
        return
    except Exception:
        try:
            from cellbin.image import cbimwrite
            cbimwrite(path, arr)
            return
        except Exception:
            import tifffile
            tifffile.imwrite(path, arr)


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

def to_semantic(mask):
    """Convert instance/binary mask to semantic uint8 0/1."""
    return (mask > 0).astype(np.uint8)


def clean_binary(mask, min_size):
    """Convert to binary and remove tiny connected components."""
    mask = (mask > 0)
    if min_size is not None and min_size > 0:
        mask = remove_small_objects(mask, min_size=min_size, connectivity=1)
    return mask.astype(np.uint8)


def compute_block_params(cols, rows, bs_x=256, bs_y=256):
    x_block_num = int(math.ceil(float(cols) / float(bs_x)))
    y_block_num = int(math.ceil(float(rows) / float(bs_y)))
    return bs_x, bs_y, x_block_num, y_block_num


def assign_cell_ids_block_sort(cx, cy, cols, rows, bs_x=256, bs_y=256):
    """
    Assign sorted cell ids.

    Rule:
      1. blkid ascending, where blkid = cx//bs_x + (cy//bs_y)*x_block_num
      2. within each block, cx descending
      3. within same cx, cy descending
      4. id starts from 0
    """
    cx = np.asarray(cx, dtype=np.int32)
    cy = np.asarray(cy, dtype=np.int32)
    n = len(cx)
    if len(cy) != n:
        raise ValueError("cx and cy length mismatch")

    if n == 0:
        return np.asarray([], dtype=np.uint32)

    bs_x, bs_y, x_block_num, y_block_num = compute_block_params(cols, rows, bs_x, bs_y)
    blkid = (cx // bs_x) + (cy // bs_y) * x_block_num

    # np.lexsort: last key is primary.
    # keys=(-cy, -cx, blkid) means blkid asc, cx desc, cy desc.
    order = np.lexsort((-cy, -cx, blkid))

    cell_ids = np.empty(n, dtype=np.uint32)
    cell_ids[order] = np.arange(n, dtype=np.uint32)
    return cell_ids


def infer_source_mode(save_path):
    """
    Decide source mode.

    Non-multimodal: save_path is None or contains none of the middle masks.
      -> all_nuclear_expand
    """
    if save_path is None:
        return "all_nuclear_expand"

    save_path = Path(save_path)
    if (save_path / "interior_mask_copy.tif").exists():
        return "interior_copy"
    if (save_path / "boundary_mask_copy.tif").exists():
        return "boundary_copy"
    if (save_path / "interior_mask_final.tif").exists() or (save_path / "output_nuclei_mask.tif").exists():
        return "default"
    return "all_nuclear_expand"


def load_source_masks(save_path, mode):
    interior_mask = None
    boundary_mask = None
    nuclei_mask_for_source = None

    if save_path is None:
        return interior_mask, boundary_mask, nuclei_mask_for_source

    save_path = Path(save_path)

    if mode == "interior_copy":
        interior_mask = to_semantic(read_tif(save_path / "interior_mask_copy.tif"))

    elif mode == "boundary_copy":
        boundary_mask = to_semantic(read_tif(save_path / "boundary_mask_copy.tif"))

    elif mode == "default":
        interior_mask_path = save_path / "interior_mask_final.tif"
        nuclei_mask_path = save_path / "output_nuclei_mask.tif"

        if interior_mask_path.exists():
            interior_mask = to_semantic(read_tif(interior_mask_path))

        if nuclei_mask_path.exists():
            nuclei_mask_for_source = to_semantic(read_tif(nuclei_mask_path))

    return interior_mask, boundary_mask, nuclei_mask_for_source


def safe_point_hit(mask, y, x):
    if mask is None:
        return False
    if y < 0 or x < 0 or y >= mask.shape[0] or x >= mask.shape[1]:
        return False
    return mask[y, x] > 0


# -----------------------------------------------------------------------------
# Main export logic
# -----------------------------------------------------------------------------

def export_cell_nuclear_geojson(
    cell_mask_path,
    nuclear_mask_path,
    out_path=None,
    save_path=None,
    min_size=15,
    bs_x=256,
    bs_y=256,
    write_relabel_tif=True,
):
    cell_mask_path = Path(cell_mask_path)
    nuclear_mask_path = Path(nuclear_mask_path)
    save_path = Path(save_path) if save_path is not None else None

    if out_path is None:
        geojson_path = cell_mask_path.with_suffix(".sorted_id.geojson")
    else:
        out_path = Path(out_path)
        if str(out_path).lower().endswith(".geojson"):
            geojson_path = out_path
            geojson_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            out_path.mkdir(parents=True, exist_ok=True)
            geojson_path = out_path / (cell_mask_path.stem + ".sorted_id.geojson")

    print("Reading cell mask:", cell_mask_path)
    cell_mask_raw = read_tif(cell_mask_path)
    final_cell_mask = clean_binary(cell_mask_raw, min_size=min_size)

    print("Reading nuclear mask:", nuclear_mask_path)
    nuclear_mask_raw = read_tif(nuclear_mask_path)
    nuclear_mask = clean_binary(nuclear_mask_raw, min_size=min_size)

    if final_cell_mask.shape != nuclear_mask.shape:
        raise ValueError(
            "cell mask shape %s != nuclear mask shape %s" %
            (str(final_cell_mask.shape), str(nuclear_mask.shape))
        )

    rows, cols = final_cell_mask.shape

    structure_4 = np.array(
        [[0, 1, 0],
         [1, 1, 1],
         [0, 1, 0]],
        dtype=np.uint8,
    )

    print("Labeling cell mask with 4-connectivity...")
    labeled_cell_mask, num_cell_labels = ndimage.label(final_cell_mask > 0, structure=structure_4)
    print("num_cell_labels =", int(num_cell_labels))

    # Collect cell centroids in label order.
    cell_props = list(regionprops(labeled_cell_mask))
    label_ids = []
    cx_list = []
    cy_list = []
    for obj in cell_props:
        label_id = int(obj.label)
        cy, cx = obj.centroid
        cx_i = int(round(cx))
        cy_i = int(round(cy))
        cx_i = int(np.clip(cx_i, 0, cols - 1))
        cy_i = int(np.clip(cy_i, 0, rows - 1))
        label_ids.append(label_id)
        cx_list.append(cx_i)
        cy_list.append(cy_i)

    sorted_cell_ids = assign_cell_ids_block_sort(
        np.asarray(cx_list, dtype=np.int32),
        np.asarray(cy_list, dtype=np.int32),
        cols=cols,
        rows=rows,
        bs_x=bs_x,
        bs_y=bs_y,
    )

    label_to_cell_id = {}
    for i, label_id in enumerate(label_ids):
        label_to_cell_id[int(label_id)] = int(sorted_cell_ids[i])

    # Build per-pixel cell_id+1 map for quick nucleus centroid lookup.
    cell_id_plus1_map = np.zeros(labeled_cell_mask.shape, dtype=np.uint32)
    for label_id, cell_id in label_to_cell_id.items():
        cell_id_plus1_map[labeled_cell_mask == label_id] = np.uint32(cell_id + 1)

    mode = infer_source_mode(save_path)
    print("GeoJSON source mode:", mode)
    interior_mask, boundary_mask, nuclei_mask_for_source = load_source_masks(save_path, mode)

    label_to_source = {}
    for obj in cell_props:
        label_id = int(obj.label)
        cy, cx = obj.centroid
        cy_i = int(np.clip(int(round(cy)), 0, rows - 1))
        cx_i = int(np.clip(int(round(cx)), 0, cols - 1))

        source = "nuclear_expand"
        if mode == "all_nuclear_expand":
            source = "nuclear_expand"
        elif mode == "interior_copy":
            source = "interior" if safe_point_hit(interior_mask, cy_i, cx_i) else "nuclear_expand"
        elif mode == "boundary_copy":
            source = "boundary" if safe_point_hit(boundary_mask, cy_i, cx_i) else "nuclear_expand"
        else:
            if safe_point_hit(interior_mask, cy_i, cx_i):
                source = "interior"
            elif safe_point_hit(nuclei_mask_for_source, cy_i, cx_i):
                source = "nuclear_expand"
            else:
                source = "boundary"

        label_to_source[label_id] = source

    transform = rasterio.transform.from_origin(0, 0, 1, 1)
    features = []

    print("Exporting cell polygons...")
    for geom, value in shapes(
        labeled_cell_mask.astype(np.int32),
        mask=(labeled_cell_mask > 0),
        transform=transform,
        connectivity=4,
    ):
        label_id = int(value)
        if label_id <= 0:
            continue
        cell_id = int(label_to_cell_id.get(label_id, -1))
        features.append({
            "type": "Feature",
            "properties": {
                "mask_type": "cell",
                "id": cell_id,
                "cell_id": cell_id,
                "cell_label": label_id,
                "source": label_to_source.get(label_id, "nuclear_expand"),
            },
            "geometry": geom,
        })

    print("Labeling nuclear mask with 4-connectivity...")
    labeled_nuclear_mask, num_nuclear_labels = ndimage.label(nuclear_mask > 0, structure=structure_4)
    print("num_nuclear_labels =", int(num_nuclear_labels))

    nucleus_cell_id_plus1_map = np.zeros(labeled_nuclear_mask.shape, dtype=np.uint32)

    print("Exporting nucleus polygons...")
    nucleus_label_to_cell_id = {}
    for obj in regionprops(labeled_nuclear_mask):
        nucleus_label = int(obj.label)
        cy, cx = obj.centroid
        cy_i = int(np.clip(int(round(cy)), 0, rows - 1))
        cx_i = int(np.clip(int(round(cx)), 0, cols - 1))

        cell_plus1 = int(cell_id_plus1_map[cy_i, cx_i])
        if cell_plus1 > 0:
            cell_id = cell_plus1 - 1
        else:
            cell_id = -1
        nucleus_label_to_cell_id[nucleus_label] = int(cell_id)

        if cell_id >= 0:
            nucleus_cell_id_plus1_map[labeled_nuclear_mask == nucleus_label] = np.uint32(cell_id + 1)

    nucleus_feature_id = 0
    for geom, value in shapes(
        labeled_nuclear_mask.astype(np.int32),
        mask=(labeled_nuclear_mask > 0),
        transform=transform,
        connectivity=4,
    ):
        nucleus_label = int(value)
        if nucleus_label <= 0:
            continue
        cell_id = int(nucleus_label_to_cell_id.get(nucleus_label, -1))
        features.append({
            "type": "Feature",
            "properties": {
                "mask_type": "nucleus",
                "id": nucleus_feature_id,
                "cell_id": cell_id,
                "nucleus_label": nucleus_label,
                "source": "nuclear",
            },
            "geometry": geom,
        })
        nucleus_feature_id += 1

    geojson = {
        "type": "FeatureCollection",
        "features": features,
    }

    print("Writing GeoJSON:", geojson_path)
    with open(str(geojson_path), "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False)

    if write_relabel_tif:
        cell_tif_path = geojson_path.with_suffix(".cell_id_plus1.tif")
        nucleus_tif_path = geojson_path.with_suffix(".nucleus_cell_id_plus1.tif")
        print("Writing relabeled cell TIFF:", cell_tif_path)
        write_tif(cell_tif_path, cell_id_plus1_map)
        print("Writing relabeled nucleus TIFF:", nucleus_tif_path)
        write_tif(nucleus_tif_path, nucleus_cell_id_plus1_map)

    print("Done.")
    print("GeoJSON:", geojson_path)
    return geojson_path


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Export sorted-id cell/nucleus GeoJSON. Python 3.8 compatible."
    )
    parser.add_argument("--cell-mask", required=True, help="final cell mask tif")
    parser.add_argument("--nuclear-mask", required=True, help="final/original nuclear mask tif")
    parser.add_argument("--save-path", default=None, help="optional middle save path for multimodal source inference")
    parser.add_argument("--out", default=None, help="output directory or .geojson path")
    parser.add_argument("--min-size", type=int, default=15, help="remove small objects smaller than this size")
    parser.add_argument("--bs-x", type=int, default=256, help="block size x for cell id sorting")
    parser.add_argument("--bs-y", type=int, default=256, help="block size y for cell id sorting")
    parser.add_argument("--relabel-tif", action="store_true", help="write relabeled uint32 TIFFs")

    args = parser.parse_args()

    export_cell_nuclear_geojson(
        cell_mask_path=args.cell_mask,
        nuclear_mask_path=args.nuclear_mask,
        out_path=args.out,
        save_path=args.save_path,
        min_size=args.min_size,
        bs_x=args.bs_x,
        bs_y=args.bs_y,
        write_relabel_tif=args.relabel_tif,
    )


if __name__ == "__main__":
    main()
