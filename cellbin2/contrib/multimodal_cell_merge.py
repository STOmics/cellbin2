#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from os.path import join
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.features import shapes
from scipy import ndimage
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects

from cellbin2.image import cbimread, cbimwrite
from cellbin2.contrib.fast_correct import run_fast_correct
from cellbin2.utils import clog


def break_diagonal_connections(binary_mask):
    """
    Break diagonal connections in a binary mask.
    Faster numpy-vectorized version.
    """
    m = (binary_mask > 0).astype(np.uint8)
    out = m.copy()

    a = m[:-1, :-1]
    b = m[:-1, 1:]
    c = m[1:, :-1]
    d = m[1:, 1:]

    # 1 0
    # 0 1
    pattern1 = (a == 1) & (b == 0) & (c == 0) & (d == 1)

    # 0 1
    # 1 0
    pattern2 = (a == 0) & (b == 1) & (c == 1) & (d == 0)

    out[1:, 1:][pattern1] = 0
    out[1:, :-1][pattern2] = 0

    return out.astype(np.uint8)


def instance2semantics(ins):
    """
    Convert instance/grayscale mask to 0/1 semantic mask.
    """
    ins = np.array(ins)
    ins[np.where(ins > 0)] = 1
    return np.array(ins, dtype=np.uint8)


def secondary_mask_filter(final_nuclear_path, final_cell_mask_path):
    if isinstance(final_nuclear_path, (str, os.PathLike)):
        final_nuclear = cbimread(final_nuclear_path, only_np=True)
    else:
        final_nuclear = final_nuclear_path

    if isinstance(final_cell_mask_path, (str, os.PathLike)):
        final_cell_mask = cbimread(final_cell_mask_path, only_np=True)
    else:
        final_cell_mask = final_cell_mask_path

    filtered_mask = np.where(final_cell_mask > 0, 0, final_nuclear)
    filtered_mask = instance2semantics(filtered_mask)
    return filtered_mask


def keep_large_nucleus_fragments(
    original_nucleus_mask: np.ndarray,
    filtered_nucleus_mask: np.ndarray,
    threshold=0.4,
) -> np.ndarray:
    """
    Keep only large remaining fragments of each original component.
    """
    original_nucleus_mask = instance2semantics(original_nucleus_mask)
    filtered_nucleus_mask = instance2semantics(filtered_nucleus_mask)

    contours, _ = cv2.findContours(
        original_nucleus_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        nucleus_roi = original_nucleus_mask[y:y + h, x:x + w]
        filtered_nucleus_roi = filtered_nucleus_mask[y:y + h, x:x + w]

        contour_roi = contour - np.array([x, y])
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(roi_mask, [contour_roi], color=1)

        not_cell_area = np.where(roi_mask != 0, 0, filtered_nucleus_roi)
        not_cell_area[np.where(not_cell_area > 0)] = 1

        nucleus_roi = cv2.bitwise_and(roi_mask, nucleus_roi)
        area = nucleus_roi.sum()

        if area == 0:
            continue

        filtered_nucleus_roi = cv2.bitwise_and(roi_mask, filtered_nucleus_roi)
        filtered_roi = label(filtered_nucleus_roi, connectivity=1)

        frag_ids = np.unique(filtered_roi)
        frag_ids = frag_ids[frag_ids != 0]

        for frag_id in frag_ids:
            frag_mask = filtered_roi == frag_id
            overlap = frag_mask.sum()
            overlap_ratio = overlap / area

            if overlap_ratio < threshold:
                filtered_roi = np.where(filtered_roi == frag_id, 0, filtered_roi)

        not_cell_area = instance2semantics(not_cell_area)
        filtered_roi = instance2semantics(filtered_roi)

        filtered_nucleus_mask[y:y + h, x:x + w] = filtered_roi + not_cell_area

    return instance2semantics(filtered_nucleus_mask)


def overlap_v3(
    secondary_mask_raw,
    primary_mask_raw,
    overlap_threshold=0.2,
    save_path="",
):
    """
    Merge secondary mask into primary mask.

    Primary has higher priority.
    Secondary only keeps fragments outside primary.
    Secondary boundary is removed before final merge.
    """
    secondary_mask_raw = secondary_mask_raw.astype(np.uint8)
    primary_mask_raw = primary_mask_raw.astype(np.uint8)

    secondary_mask = instance2semantics(secondary_mask_raw)
    primary_mask = instance2semantics(primary_mask_raw)

    filtered_secondary_mask = secondary_mask_filter(secondary_mask, primary_mask)

    secondary_mask_final = keep_large_nucleus_fragments(
        secondary_mask,
        filtered_secondary_mask,
        threshold=overlap_threshold,
    )

    contours, _ = cv2.findContours(
        secondary_mask_final,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    secondary_boundary = np.zeros_like(secondary_mask_final)
    cv2.drawContours(secondary_boundary, contours, -1, 1, 1)

    primary_mask_add_secondary = np.add(primary_mask, secondary_mask_final)

    save_primary_mask = np.where(
        secondary_boundary > 0,
        0,
        primary_mask_add_secondary,
    )
    secondary_mask_final = np.where(
        secondary_boundary > 0,
        0,
        secondary_mask_final,
    )

    return instance2semantics(secondary_mask_final), instance2semantics(save_primary_mask)


def interior_filter(interior_mask: np.ndarray, nuclei_mask: np.ndarray) -> np.ndarray:
    """
    Remove interior components with too little nucleus overlap.
    """
    interior_mask = instance2semantics(interior_mask)
    nuclei_mask = instance2semantics(nuclei_mask)

    contours, _ = cv2.findContours(
        interior_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        nuclei_roi = nuclei_mask[y:y + h, x:x + w]

        contour_roi = contour - np.array([x, y])
        roi_mask = np.zeros(shape=(h, w), dtype=np.uint8)
        cv2.fillPoly(roi_mask, pts=[contour_roi], color=1)

        interior_and_nuclei = cv2.bitwise_and(roi_mask, nuclei_roi)

        total_area = np.sum(roi_mask > 0)
        if total_area > 0:
            overlap_ratio = np.sum(interior_and_nuclei > 0) / total_area
            if overlap_ratio < 0.1:
                cv2.fillPoly(interior_mask, pts=[contour], color=0)

    return instance2semantics(interior_mask)


def export_cell_mask_to_geojson(
    final_cell_mask_path,
    final_nuclear_path,
    save_path,
):
    """
    Export final cell mask and original nuclear mask into one GeoJSON.

    Cell polygons:
        mask_type = cell
        source = boundary / interior / nuclear_expand

    Nucleus polygons:
        mask_type = nucleus
        source = nuclear
    """
    final_cell_mask_path = Path(final_cell_mask_path)
    final_nuclear_path = Path(final_nuclear_path)
    save_path = Path(save_path)

    geojson_path = final_cell_mask_path.with_suffix(".geojson")

    # 1. Read and clean final cell mask.
    final_cell_mask = cbimread(final_cell_mask_path, only_np=True)
    final_cell_mask = remove_small_objects(
        final_cell_mask.astype(bool),
        min_size=15,
        connectivity=1,
    ).astype(np.uint8)
    cbimwrite(final_cell_mask_path, final_cell_mask * 255)

    # 2. Read and clean original nuclear mask.
    nuclear_mask = None
    if final_nuclear_path.exists():
        nuclear_mask = cbimread(final_nuclear_path, only_np=True)
        nuclear_mask = remove_small_objects(
            nuclear_mask.astype(bool),
            min_size=15,
            connectivity=1,
        ).astype(np.uint8)

    # 3. Label with 4-connectivity.
    structure_4 = np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ],
        dtype=np.uint8,
    )

    labeled_cell_mask, num_cell_labels = ndimage.label(
        final_cell_mask > 0,
        structure=structure_4,
    )

    # 4. Decide GeoJSON source mode.
    #
    # 三 mask 默认模式:
    #   interior_mask_final.tif + output_nuclei_mask.tif
    #
    # 双 mask + mem:
    #   boundary_mask_copy.tif
    #
    # 双 mask + cyto:
    #   interior_mask_copy.tif
    if (save_path / "interior_mask_copy.tif").exists():
        mode = "interior_copy"
    elif (save_path / "boundary_mask_copy.tif").exists():
        mode = "boundary_copy"
    elif (save_path / "interior_mask_final.tif").exists() or (
        save_path / "output_nuclei_mask.tif"
    ).exists():
        mode = "default"
    else:
        mode = "all_nuclear_expand"

    clog.info(f"GeoJSON source mode: {mode}, num_cell_labels={num_cell_labels}")

    interior_mask = None
    boundary_mask = None
    nuclei_mask_for_source = None

    if mode == "interior_copy":
        interior_mask = cbimread(save_path / "interior_mask_copy.tif", only_np=True)
        interior_mask = instance2semantics(interior_mask)

    elif mode == "boundary_copy":
        boundary_mask = cbimread(save_path / "boundary_mask_copy.tif", only_np=True)
        boundary_mask = instance2semantics(boundary_mask)

    elif mode == "default":
        interior_mask_path = save_path / "interior_mask_final.tif"
        nuclei_mask_path = save_path / "output_nuclei_mask.tif"

        if interior_mask_path.exists():
            interior_mask = cbimread(interior_mask_path, only_np=True)
            interior_mask = instance2semantics(interior_mask)

        if nuclei_mask_path.exists():
            nuclei_mask_for_source = cbimread(nuclei_mask_path, only_np=True)
            nuclei_mask_for_source = instance2semantics(nuclei_mask_for_source)

    # 5. Infer cell source by centroid.
    label_to_source = {}
    h, w = labeled_cell_mask.shape

    for obj in regionprops(labeled_cell_mask):
        label_id = int(obj.label)

        cy, cx = obj.centroid
        cy = int(round(cy))
        cx = int(round(cx))
        cy = int(np.clip(cy, 0, h - 1))
        cx = int(np.clip(cx, 0, w - 1))

        source = "nuclear_expand"

        if mode == "all_nuclear_expand":
            source = "nuclear_expand"

        elif mode == "interior_copy":
            if (
                interior_mask is not None
                and cy < interior_mask.shape[0]
                and cx < interior_mask.shape[1]
                and interior_mask[cy, cx] > 0
            ):
                source = "interior"
            else:
                source = "nuclear_expand"

        elif mode == "boundary_copy":
            if (
                boundary_mask is not None
                and cy < boundary_mask.shape[0]
                and cx < boundary_mask.shape[1]
                and boundary_mask[cy, cx] > 0
            ):
                source = "boundary"
            else:
                source = "nuclear_expand"

        else:
            if (
                interior_mask is not None
                and cy < interior_mask.shape[0]
                and cx < interior_mask.shape[1]
                and interior_mask[cy, cx] > 0
            ):
                source = "interior"
            elif (
                nuclei_mask_for_source is not None
                and cy < nuclei_mask_for_source.shape[0]
                and cx < nuclei_mask_for_source.shape[1]
                and nuclei_mask_for_source[cy, cx] > 0
            ):
                source = "nuclear_expand"
            else:
                source = "boundary"

        label_to_source[label_id] = source

    transform = rasterio.transform.from_origin(0, 0, 1, 1)
    features = []

    # 6. Export final cell polygons.
    for geom, value in shapes(
        labeled_cell_mask.astype(np.int32),
        mask=(labeled_cell_mask > 0),
        transform=transform,
        connectivity=4,
    ):
        label_id = int(value)
        if label_id <= 0:
            continue

        features.append(
            {
                "type": "Feature",
                "properties": {
                    "mask_type": "cell",
                    "source": label_to_source.get(label_id, "nuclear_expand"),
                },
                "geometry": geom,
            }
        )

    # 7. Export original nucleus polygons.
    if nuclear_mask is not None:
        labeled_nuclear_mask, num_nuclear_labels = ndimage.label(
            nuclear_mask > 0,
            structure=structure_4,
        )
        clog.info(f"num_nuclear_labels={num_nuclear_labels}")

        for geom, value in shapes(
            labeled_nuclear_mask.astype(np.int32),
            mask=(labeled_nuclear_mask > 0),
            transform=transform,
            connectivity=4,
        ):
            label_id = int(value)
            if label_id <= 0:
                continue

            features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "mask_type": "nucleus",
                        "source": "nuclear",
                    },
                    "geometry": geom,
                }
            )

    geojson = {
        "type": "FeatureCollection",
        "features": features,
    }

    with open(geojson_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False)

    clog.info(f"Saved GeoJSON to: {geojson_path}")


def multimodal_merge(
    nuclei_mask_raw,
    cell_mask_raw,
    interior_mask_raw,
    overlap_threshold=0.5,
    save_path="",
    expand_distance=10,
    expand_n_jobs=30,
    final_overlap_threshold=0.1,
):
    """
    Three-mask mode:
        nuclei + membrane/boundary + cyto/interior
    """
    # 1. Merge interior into membrane/cell.
    interior_mask_final, cell_add_interior = overlap_v3(
        interior_mask_raw,
        cell_mask_raw,
        overlap_threshold=overlap_threshold,
        save_path=save_path,
    )

    interior_mask_final = instance2semantics(interior_mask_final)
    nuclei_mask_semantic = instance2semantics(nuclei_mask_raw)

    filter_mask = interior_filter(interior_mask_final, nuclei_mask_semantic)
    cell_add_interior = cv2.bitwise_or(
        instance2semantics(cell_mask_raw),
        instance2semantics(filter_mask),
    )

    if save_path != "":
        cbimwrite(join(save_path, "interior_mask_final.tif"), interior_mask_final * 255)
        cbimwrite(
            join(save_path, "cell_mask_add_interior.tif"),
            instance2semantics(cell_add_interior) * 255,
        )

    # 2. First merge nuclei with cell/interior.
    output_nuclei_mask, first_merged_mask = overlap_v3(
        nuclei_mask_raw,
        cell_add_interior,
        overlap_threshold=0.8,
        save_path=save_path,
    )

    first_merged_mask = instance2semantics(first_merged_mask)

    if save_path != "":
        output_nuclei_path = join(save_path, "output_nuclei_mask.tif")
        cbimwrite(output_nuclei_path, instance2semantics(output_nuclei_mask) * 255)
        cbimwrite(join(save_path, "merged_cell_mask.tif"), first_merged_mask * 255)
    else:
        output_nuclei_path = None

    # 3. Expand remaining nuclei.
    if output_nuclei_path is not None and os.path.exists(output_nuclei_path):
        fast_mask = run_fast_correct(
            mask_path=output_nuclei_path,
            distance=expand_distance,
            n_jobs=expand_n_jobs,
        )
    else:
        fast_mask = output_nuclei_mask

    if save_path != "":
        cbimwrite(join(save_path, "expand_nuclei.tif"), fast_mask)

    # 4. Second merge expanded nuclei with cell/interior.
    secondary_mask_final, final_mask = overlap_v3(
        fast_mask,
        cell_add_interior,
        overlap_threshold=final_overlap_threshold,
        save_path="",
    )

    final_mask = instance2semantics(final_mask).astype(np.uint8)

    if save_path != "":
        cbimwrite(
            join(save_path, "secondary_mask_final.tif"),
            instance2semantics(secondary_mask_final) * 255,
        )
        cbimwrite(join(save_path, "final_cell_mask.tif"), final_mask * 255)

    return final_mask


def run_dual_modal(
    nuclei_mask_path: str,
    secondary_mask_path: str,
    secondary_source: str,
    save_path: str,
):
    """
    Two-mask mode.

    secondary_source:
        "boundary" for --mem
        "interior" for --cyto
    """
    nuclei_mask = cbimread(nuclei_mask_path, only_np=True)
    secondary_mask = cbimread(secondary_mask_path, only_np=True)

    secondary_mask = instance2semantics(secondary_mask)

    # Save a copy of the second input mask so GeoJSON can infer source.
    if secondary_source == "boundary":
        cbimwrite(
            join(save_path, "boundary_mask_copy.tif"),
            secondary_mask * 255,
        )
    elif secondary_source == "interior":
        cbimwrite(
            join(save_path, "interior_mask_copy.tif"),
            secondary_mask * 255,
        )
    else:
        raise ValueError(f"Unknown secondary_source: {secondary_source}")

    # 1. Remove nuclei already covered by secondary mask.
    output_nuclei_mask, _ = overlap_v3(
        nuclei_mask,
        secondary_mask,
        overlap_threshold=0.8,
        save_path=save_path,
    )

    output_nuclei_path = os.path.join(save_path, "output_nuclei_mask.tif")
    cbimwrite(output_nuclei_path, instance2semantics(output_nuclei_mask) * 255)

    # 2. Expand remaining nuclei.
    fast_mask = run_fast_correct(
        mask_path=output_nuclei_path,
        distance=10,
        n_jobs=30,
    )

    expand_nuclei_path = os.path.join(save_path, "expand_nuclei.tif")
    cbimwrite(expand_nuclei_path, fast_mask)

    # 3. Merge expanded nuclei into secondary mask.
    secondary_mask_final, final_mask = overlap_v3(
        fast_mask,
        secondary_mask,
        overlap_threshold=0.1,
        save_path="",
    )

    final_mask = instance2semantics(final_mask).astype(np.uint8)

    cbimwrite(
        join(save_path, "secondary_mask_final.tif"),
        instance2semantics(secondary_mask_final) * 255,
    )
    cbimwrite(join(save_path, "final_cell_mask.tif"), final_mask * 255)


def parse_args():
    parser = argparse.ArgumentParser(description="Distributed Mask Fusion")

    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output directory for merged results.",
    )
    parser.add_argument(
        "--nuc",
        required=True,
        help="Path to nucleus / DAPI mask file.",
    )
    parser.add_argument(
        "--mem",
        default=None,
        help="Path to membrane / CY5 mask file.",
    )
    parser.add_argument(
        "--cyto",
        default=None,
        help="Path to interior / TRITC mask file.",
    )

    return parser.parse_args()


def ensure_output_dir(output_dir: str) -> str:
    abs_output_path = os.path.abspath(output_dir)
    os.makedirs(abs_output_path, exist_ok=True)
    return abs_output_path


def run_pipeline(args):
    save_path = args.output
    nuclei_mask_path = args.nuc
    cell_mask_path = args.mem
    interior_mask_path = args.cyto

    has_interior = interior_mask_path is not None
    has_boundary = cell_mask_path is not None

    print(">>> Starting Distributed Mask Fusion")
    print(f"Target Nuclei   (--nuc):  {nuclei_mask_path}")
    print(f"Target Membrane (--mem):  {cell_mask_path}")
    print(f"Target Interior (--cyto): {interior_mask_path}")

    nuclei_mask = cbimread(nuclei_mask_path, only_np=True)

    if has_interior and has_boundary:
        cell_mask = cbimread(cell_mask_path, only_np=True)
        interior_mask = cbimread(interior_mask_path, only_np=True)

        multimodal_merge(
            nuclei_mask_raw=nuclei_mask,
            cell_mask_raw=cell_mask,
            interior_mask_raw=interior_mask,
            save_path=save_path,
        )

    elif has_boundary:
        run_dual_modal(
            nuclei_mask_path=nuclei_mask_path,
            secondary_mask_path=cell_mask_path,
            secondary_source="boundary",
            save_path=save_path,
        )

    elif has_interior:
        run_dual_modal(
            nuclei_mask_path=nuclei_mask_path,
            secondary_mask_path=interior_mask_path,
            secondary_source="interior",
            save_path=save_path,
        )

    else:
        raise ValueError(
            "Invalid input: nucleus mask is required, and at least one of "
            "membrane/interior mask must be provided."
        )

    # Auto export GeoJSON.
    export_cell_mask_to_geojson(
        final_cell_mask_path=os.path.join(save_path, "final_cell_mask.tif"),
        final_nuclear_path=nuclei_mask_path,
        save_path=Path(save_path),
    )


def main():
    args = parse_args()
    abs_output_path = ensure_output_dir(args.output)

    run_pipeline(args)

    print(f"\n[Success] Final Merged Mask: {os.path.join(abs_output_path, 'final_cell_mask.tif')}")
    print(f"[Success] Final GeoJSON:     {os.path.join(abs_output_path, 'final_cell_mask.geojson')}")


if __name__ == "__main__":
    main()