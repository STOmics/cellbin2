#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Export cell mask + nuclear mask to GeoJSON with sorted cell_id.

Compatible with Python >= 3.8.

Features
--------
1. final_cell_mask cell_ids are assigned by the geftools snake-scan +
   fixed-seed shuffle algorithm (cellbin2.utils.cell_id_sort), replicating
   cgefCellgem.cpp rebuildCellIdMapForSubset:
   - 4-connectivity connected components (matches original cell-mask connectivity)
   - snake scan (even rows x asc, odd rows x desc) over (y, label)-sorted cells
   - fixed-seed (2193796729) Fisher-Yates shuffle
   - cell_id is dense 1..N (1-based)
2. final_nuclear_mask is assigned to cell_id by nucleus centroid falling inside cell mask.
   Multiple nuclei can share the same cell_id.
3. Compatible with multimodal and non-multimodal modes:
   - If --save-path is omitted or contains no middle masks, all cell source = nuclear_expand.
   - If middle masks exist, infer source from interior/boundary/output_nuclei masks.
4. GeoJSON polygon coordinates are converted from float to int.
5. Only GeoJSON is written. No relabeled TIFF output.

Example
-------
python cell_nuclear_geojson.py \
  --cell-mask final_cell_mask.tif \
  --nuclear-mask final_nuclear_mask.tif \
  --save-path /path/to/middle_masks \
  --out /path/to/output_dir
"""

from __future__ import print_function

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage
from skimage.measure import regionprops
from skimage.morphology import remove_small_objects
from rasterio.features import shapes
import rasterio.transform
from rasterio.transform import Affine


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

    if y < 0 or x < 0:
        return False

    if y >= mask.shape[0] or x >= mask.shape[1]:
        return False

    return mask[y, x] > 0


def geometry_coords_to_int(geometry):
    """
    Recursively convert all GeoJSON coordinate values to Python int.

    rasterio.features.shapes returns coordinates such as:
        [123.0, -456.0]

    This function converts them to:
        [123, -456]
    """

    def convert_coords(coords):
        if isinstance(coords, (list, tuple)):
            # A coordinate point, such as [x, y].
            if (
                len(coords) >= 2
                and all(
                    isinstance(
                        value,
                        (
                            int,
                            float,
                            np.integer,
                            np.floating,
                        ),
                    )
                    for value in coords
                )
            ):
                return [
                    int(value)
                    for value in coords
                ]

            # Nested rings / polygons / multipolygons.
            return [
                convert_coords(item)
                for item in coords
            ]

        return coords

    geometry_int = dict(geometry)
    geometry_int["coordinates"] = convert_coords(
        geometry["coordinates"]
    )

    return geometry_int


# -----------------------------------------------------------------------------
# Main export logic
# -----------------------------------------------------------------------------

def export_cell_nuclear_geojson(
    cell_mask_path,
    nuclear_mask_path,
    out_path=None,
    save_path=None,
    min_size=15,
):
    cell_mask_path = Path(cell_mask_path)
    nuclear_mask_path = Path(nuclear_mask_path)
    save_path = (
        Path(save_path)
        if save_path not in (None, "")
        else None
    )

    if out_path is None:
        geojson_path = cell_mask_path.with_suffix(
            ".sorted_id.geojson"
        )
    else:
        out_path = Path(out_path)

        if str(out_path).lower().endswith(".geojson"):
            geojson_path = out_path
            geojson_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
        else:
            out_path.mkdir(
                parents=True,
                exist_ok=True,
            )
            geojson_path = out_path / (
                cell_mask_path.stem
                + ".sorted_id.geojson"
            )

    print("Reading cell mask:", cell_mask_path)
    cell_mask_raw = read_tif(cell_mask_path)
    final_cell_mask = clean_binary(
        cell_mask_raw,
        min_size=min_size,
    )

    print("Reading nuclear mask:", nuclear_mask_path)
    nuclear_mask_raw = read_tif(nuclear_mask_path)
    nuclear_mask = clean_binary(
        nuclear_mask_raw,
        min_size=min_size,
    )

    if final_cell_mask.shape != nuclear_mask.shape:
        raise ValueError(
            "cell mask shape %s != nuclear mask shape %s"
            % (
                str(final_cell_mask.shape),
                str(nuclear_mask.shape),
            )
        )

    rows, cols = final_cell_mask.shape

    structure_4 = np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ],
        dtype=np.uint8,
    )

    # ---- Cell ID assignment: geftools snake-scan + fixed-seed shuffle ----
    # Replicates cgefCellgem.cpp rebuildCellIdMapForSubset: 4-connectivity CC
    # labeling (matches the original cell-mask connectivity), snake-scan order,
    # fixed-seed (2193796729) Fisher-Yates shuffle -> dense 1..N cell_id
    # (1-based). This MUST match the pipeline's cell numbering so GeoJSON
    # cell_ids align with the gef/cellbin ids.
    # NOTE: apply_contour_filter=False because the module requires connectivity=8
    # for the contour filter (cv2.findContours is 8-connected); 4-connectivity is
    # kept to preserve the original cell set. Cell cleaning is via min_size only.
    from cellbin2.utils.cell_id_sort import (
        build_cell_id_map_from_mask,
        apply_cell_ids_to_mask,
    )

    print("Assigning cell ids (snake-scan + shuffle, 4-connectivity)...")
    id_map, cc_label_map = build_cell_id_map_from_mask(
        final_cell_mask,
        apply_contour_filter=False,
        connectivity=4,
    )
    # pixel = cell_id (1-based, dense 1..N); 0 = background or contour-filtered.
    cell_id_map = apply_cell_ids_to_mask(cc_label_map, id_map)
    num_cells = len(id_map)
    print("num_cells =", num_cells)

    mode = infer_source_mode(save_path)
    print("GeoJSON source mode:", mode)

    (
        interior_mask,
        boundary_mask,
        nuclei_mask_for_source,
    ) = load_source_masks(save_path, mode)

    # ---- per-cell source classification (interior / boundary / nuclear_expand) ----
    # regionprops on cell_id_map: each obj.label IS the (1-based) cell_id.
    cell_id_to_source = {}

    for obj in regionprops(cell_id_map):
        cell_id = int(obj.label)
        cy, cx = obj.centroid

        cy_i = int(
            np.clip(
                int(round(cy)),
                0,
                rows - 1,
            )
        )
        cx_i = int(
            np.clip(
                int(round(cx)),
                0,
                cols - 1,
            )
        )

        source = "nuclear_expand"

        if mode == "all_nuclear_expand":
            source = "nuclear_expand"

        elif mode == "interior_copy":
            source = (
                "interior"
                if safe_point_hit(
                    interior_mask,
                    cy_i,
                    cx_i,
                )
                else "nuclear_expand"
            )

        elif mode == "boundary_copy":
            source = (
                "boundary"
                if safe_point_hit(
                    boundary_mask,
                    cy_i,
                    cx_i,
                )
                else "nuclear_expand"
            )

        else:
            if safe_point_hit(
                interior_mask,
                cy_i,
                cx_i,
            ):
                source = "interior"

            elif safe_point_hit(
                nuclei_mask_for_source,
                cy_i,
                cx_i,
            ):
                source = "nuclear_expand"

            else:
                source = "boundary"

        cell_id_to_source[cell_id] = source

    transform = Affine(1, 0, 0, 0, 1, 0)

    features = []

    print("Exporting cell polygons...")

    # shapes on the cell_id map: each distinct value (cell_id) is one polygon,
    # so value IS the cell_id directly — no label lookup needed. Filtered
    # cells (cell_id == 0) are excluded by the mask.
    for geom, value in shapes(
        cell_id_map.astype(np.int32),
        mask=(cell_id_map > 0),
        transform=transform,
        connectivity=4,
    ):
        cell_id = int(value)

        if cell_id <= 0:
            continue

        features.append(
            {
                "type": "Feature",
                "properties": {
                    "mask_type": "cell",
                    "id": cell_id,
                    "cell_id": cell_id,
                    "source": cell_id_to_source.get(
                        cell_id,
                        "nuclear_expand",
                    ),
                },
                "geometry": geometry_coords_to_int(
                    geom
                ),
            }
        )

    print("Labeling nuclear mask with 4-connectivity...")

    (
        labeled_nuclear_mask,
        num_nuclear_labels,
    ) = ndimage.label(
        nuclear_mask > 0,
        structure=structure_4,
    )

    print(
        "num_nuclear_labels =",
        int(num_nuclear_labels),
    )

    print("Assigning nucleus to cell_id...")

    # nucleus connected-component label -> parent cell_id
    # cell_id_map is 1-based (0 = background / no cell), so a centroid hit
    # returns the cell_id directly and 0 means "miss -> overlap fallback".
    nucleus_label_to_cell_id = {}

    for obj in regionprops(labeled_nuclear_mask):
        nucleus_label = int(obj.label)
        cy, cx = obj.centroid
        cy_i = int(np.clip(int(round(cy)), 0, rows - 1))
        cx_i = int(np.clip(int(round(cx)), 0, cols - 1))

        # First try the nucleus centroid.
        cell_id = int(cell_id_map[cy_i, cx_i])

        if cell_id <= 0:
            # Fallback: choose the cell overlapping the most nucleus pixels.
            coords = obj.coords
            overlap = cell_id_map[
                coords[:, 0],
                coords[:, 1],
            ]
            overlap = overlap[
                overlap > 0
            ]

            if overlap.size > 0:
                unique_ids, counts = np.unique(
                    overlap,
                    return_counts=True,
                )
                cell_id = int(unique_ids[np.argmax(counts)])
            else:
                # 1-based scheme: 0 = no parent cell.
                cell_id = 0

        nucleus_label_to_cell_id[nucleus_label] = int(cell_id)

    print("Exporting nucleus polygons...")

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

        cell_id = int(
            nucleus_label_to_cell_id.get(
                nucleus_label,
                0,
            )
        )

        features.append(
            {
                "type": "Feature",
                "properties": {
                    "mask_type": "nucleus",
                    "id": nucleus_feature_id,
                    "cell_id": cell_id,
                    "source": "nuclear",
                },
                "geometry": geometry_coords_to_int(
                    geom
                ),
            }
        )

        nucleus_feature_id += 1

    geojson = {
        "type": "FeatureCollection",
        "features": features,
    }

    print("Writing GeoJSON:", geojson_path)

    with open(
        str(geojson_path),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            geojson,
            f,
            ensure_ascii=False,
        )

    print("Done.")
    print("GeoJSON:", geojson_path)

    return geojson_path


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Export sorted-id cell/nucleus GeoJSON "
            "with integer polygon coordinates. "
            "Python 3.8 compatible."
        )
    )

    parser.add_argument(
        "--cell-mask",
        required=True,
        help="final cell mask tif",
    )

    parser.add_argument(
        "--nuclear-mask",
        required=True,
        help="final/original nuclear mask tif",
    )

    parser.add_argument(
        "--save-path",
        default=None,
        help=(
            "optional middle save path "
            "for multimodal source inference"
        ),
    )

    parser.add_argument(
        "--out",
        default=None,
        help="output directory or .geojson path",
    )

    parser.add_argument(
        "--min-size",
        type=int,
        default=15,
        help="remove small objects smaller than this size",
    )

    args = parser.parse_args()

    export_cell_nuclear_geojson(
        cell_mask_path=args.cell_mask,
        nuclear_mask_path=args.nuclear_mask,
        out_path=args.out,
        save_path=args.save_path,
        min_size=args.min_size,
    )


if __name__ == "__main__":
    main()
