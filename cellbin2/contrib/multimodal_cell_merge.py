import os
from os.path import join
from typing import Final, NamedTuple, TypedDict, Tuple

from skimage.measure import label
import numpy as np
from scipy import ndimage
import cv2

from cellbin2.image import cbimread, cbimwrite
from cellbin2.contrib.fast_correct import run_fast_correct
import json
from pathlib import Path

import rasterio
from rasterio.features import shapes
from skimage.measure import regionprops
from skimage.morphology import remove_small_objects
from cellbin2.utils import clog



def break_diagonal_connections(binary_mask):
    """
    Break diagonal connections in a binary mask.
    Faster numpy-vectorized version.
    """
    m = (binary_mask > 0).astype(np.uint8)
    out = m.copy()

    a = m[:-1, :-1]   # top-left
    b = m[:-1, 1:]    # top-right
    c = m[1:, :-1]    # bottom-left
    d = m[1:, 1:]     # bottom-right

    # 1 0
    # 0 1
    pattern1 = (a == 1) & (b == 0) & (c == 0) & (d == 1)

    # 0 1
    # 1 0
    pattern2 = (a == 0) & (b == 1) & (c == 1) & (d == 0)

    # 对应删除位置
    out[1:, 1:][pattern1] = 0
    out[1:, :-1][pattern2] = 0

    return out
def export_cell_mask_to_geojson(final_cell_mask_path, save_path):
    final_cell_mask_path = Path(final_cell_mask_path)
    save_path = Path(save_path)
    geojson_path = final_cell_mask_path.with_suffix(".geojson")

    # 1) read and clean cell mask
    final_cell_mask = cbimread(final_cell_mask_path, only_np=True)
    final_cell_mask = remove_small_objects(
        final_cell_mask.astype(np.bool8),
        min_size=15,
        connectivity=1
    ).astype(np.uint8)

    # keep original behavior
    cbimwrite(final_cell_mask_path, final_cell_mask)

    # 2) 4-connectivity labeling
    structure_4 = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
    ], dtype=np.uint8)
    labeled_mask, num_labels = ndimage.label(final_cell_mask > 0, structure=structure_4)

    # 3) decide mode
    if (not save_path.exists()) or (not any(save_path.iterdir())):
        mode = "all_nuclear"
    elif (save_path / "interior_mask_copy.tif").exists():
        mode = "interior_copy"
    elif (save_path / "boundary_mask_copy.tif").exists():
        mode = "boundary_copy"
    else:
        mode = "default"

    clog.info(f"GeoJSON source mode: {mode}, num_labels={num_labels}")

    # 4) read optional masks
    interior_mask = None
    boundary_mask = None
    nuclei_mask = None

    if mode == "interior_copy":
        interior_mask_path = save_path / "interior_mask_copy.tif"
        interior_mask = cbimread(interior_mask_path, only_np=True)
        interior_mask = (interior_mask > 0).astype(np.uint8)

    elif mode == "boundary_copy":
        boundary_mask_path = save_path / "boundary_mask_copy.tif"
        boundary_mask = cbimread(boundary_mask_path, only_np=True)
        boundary_mask = (boundary_mask > 0).astype(np.uint8)

    elif mode == "default":
        interior_mask_path = save_path / "interior_mask_final.tif"
        nuclei_mask_path = save_path / "output_nuclei_mask.tif"

        if interior_mask_path.exists():
            interior_mask = cbimread(interior_mask_path, only_np=True)
            interior_mask = (interior_mask > 0).astype(np.uint8)

        if nuclei_mask_path.exists():
            nuclei_mask = cbimread(nuclei_mask_path, only_np=True)
            nuclei_mask = (nuclei_mask > 0).astype(np.uint8)

    # 5) compute source for each label by centroid
    label_to_source = {}
    h, w = labeled_mask.shape

    props = regionprops(labeled_mask)
    for obj in props:
        label_id = int(obj.label)

        cy, cx = obj.centroid
        cy = int(round(cy))
        cx = int(round(cx))
        cy = int(np.clip(cy, 0, h - 1))
        cx = int(np.clip(cx, 0, w - 1))

        source = "nuclear"

        if mode == "all_nuclear":
            source = "nuclear"

        elif mode == "interior_copy":
            if (
                interior_mask is not None
                and cy < interior_mask.shape[0]
                and cx < interior_mask.shape[1]
                and interior_mask[cy, cx] > 0
            ):
                source = "interior"
            else:
                source = "nuclear"

        elif mode == "boundary_copy":
            if (
                boundary_mask is not None
                and cy < boundary_mask.shape[0]
                and cx < boundary_mask.shape[1]
                and boundary_mask[cy, cx] > 0
            ):
                source = "boundary"
            else:
                source = "nuclear"

        else:
            if (
                interior_mask is not None
                and cy < interior_mask.shape[0]
                and cx < interior_mask.shape[1]
                and interior_mask[cy, cx] > 0
            ):
                source = "interior"
            elif (
                nuclei_mask is not None
                and cy < nuclei_mask.shape[0]
                and cx < nuclei_mask.shape[1]
                and nuclei_mask[cy, cx] > 0
            ):
                source = "nuclear"
            else:
                source = "boundary"

        label_to_source[label_id] = source

    # 6) use pixel coordinates directly
    transform = rasterio.transform.from_origin(0, 0, 1, 1)

    # 7) polygonize whole labeled mask once
    features = []
    for geom, value in shapes(
        labeled_mask.astype(np.int32),
        mask=(labeled_mask > 0),
        transform=transform,
        connectivity=4
    ):
        label_id = int(value)
        if label_id <= 0:
            continue

        features.append({
            "type": "Feature",
            "properties": {
                "source": label_to_source.get(label_id, "nuclear")
            },
            "geometry": geom
        })

    geojson = {
        "type": "FeatureCollection",
        "features": features
    }

    with open(geojson_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False)

    clog.info(f"Saved GeoJSON to: {geojson_path}")

MAX_INPUT_LABEL_VALUE: Final[int] = np.iinfo(np.uint32).max



# @process_decorator('GiB')
def unique_nonzero_pairs_numpy(masks):
    """Compute the unique pairs between to labeled masks with nonzero labels using numpy.

    Args:
        masks (tuple[np.ndarray, np.ndarray]): The masks to compare and
            generated unique pairings.

    Returns:
        np.ndarray[tuple[int, int], np.intp]: A matrix of shape `(p, 2)`
            containing the `p` unique pairs.
        np.ndarray[tuple[int, int], int]: An array of shape `(p,)` of counts
            specifying how many times each pair occurred.
    """
    mask0 = masks[0].ravel()
    mask1 = masks[1].ravel()
    # Find pairs where both labels are non-zero
    valid_indices = (mask0 > 0) & (mask1 > 0)
    valid_mask0 = mask0[valid_indices]
    valid_mask1 = mask1[valid_indices]
    # Combine valid pairs
    combined = np.column_stack((valid_mask0, valid_mask1))
    # Find unique pairs and their counts
    unique_pairs, counts = np.unique(combined, axis=0, return_counts=True)
    return unique_pairs, counts



def keep_large_nucleus_fragments(original_nucleus_mask: np.ndarray, filtered_nucleus_mask: np.ndarray, threshold=0.4) -> np.ndarray:
    """
    keep only big pieces of cell pieces
    """
    original_nucleus_mask = instance2semantics(original_nucleus_mask)
    filtered_nucleus_mask = instance2semantics(filtered_nucleus_mask)
    result_mask = np.zeros_like(filtered_nucleus_mask, dtype=np.uint8)
    contours, _ = cv2.findContours(original_nucleus_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    for contour in contours:

        x, y, w, h = cv2.boundingRect(contour)
        nucleus_roi = original_nucleus_mask[y:y+h, x:x+w]
        filtered_nucleus_roi = filtered_nucleus_mask[y:y+h, x:x+w]
        contour_roi = contour - np.array([x,y])
        roi_mask = np.zeros((h,w), dtype=np.uint8)
        
        cv2.fillPoly(roi_mask, [contour_roi],(1,)) # cell before filterd 
        not_cell_area = np.where(roi_mask != 0, 0,filtered_nucleus_roi)
        not_cell_area[np.where(not_cell_area > 0)] = 1
        nucleus_roi = cv2.bitwise_and(roi_mask, nucleus_roi)
        area = nucleus_roi.sum()
        filtered_nucleus_roi = cv2.bitwise_and(roi_mask, filtered_nucleus_roi)

        filtered_roi = label(filtered_nucleus_roi, connectivity=1)
        frag_ids = np.unique(filtered_roi)
        frag_ids = frag_ids[frag_ids != 0]
        for frag_id in frag_ids:
            frag_mask = (filtered_roi == frag_id)
            overlap = frag_mask.sum()
            
            overlap_ratio = overlap / area
            if overlap_ratio < threshold:
                # remove small piece of cell
                filtered_roi = np.where(filtered_roi == frag_id, 0,filtered_roi)
                
        not_cell_area = instance2semantics(not_cell_area)
        filtered_roi = instance2semantics(filtered_roi)
        filtered_nucleus_mask[y:y+h, x:x+w] = filtered_roi + not_cell_area
    return filtered_nucleus_mask

def cell_filter(final_nuclear_path,final_cell_mask_path):
    final_nuclear = cbimread(final_nuclear_path, only_np=True)
    final_cell_mask = cbimread(final_cell_mask_path, only_np=True)
    filtered_mask = final_nuclear * final_cell_mask
    filtered_mask = instance2semantics(filtered_mask)
    return filtered_mask

def secondary_mask_filter(final_nuclear_path,final_cell_mask_path):
    if isinstance(final_nuclear_path, (str, os.PathLike, np.ndarray)):
        final_nuclear = cbimread(final_nuclear_path, only_np=True)
    else:
        final_nuclear = final_nuclear_path
    if isinstance(final_cell_mask_path, (str, os.PathLike, np.ndarray)):   
        final_cell_mask = cbimread(final_cell_mask_path, only_np=True)
    else:
        final_cell_mask = final_cell_mask_path
    filtered_mask = np.where(final_cell_mask > 0, 0, final_nuclear)
    filtered_mask = instance2semantics(filtered_mask)
    return filtered_mask

# @process_decorator('GiB')
def make_mask_consecutive(
        mask,
        start_from: int = 1,
):
    """Given a mask of integers, reassign the labels to be consecutive.

    Args:
        mask (np.ndarray[tuple[int, int], np.uint32]): a mask of positive integers,
        with 0 meaning background, which might not be consecutive

    Returns:
        mask: a new mask where the labels are consecutive integers
    """
    unique_input_labels = np.unique(mask) 
    unique_input_labels = unique_input_labels[unique_input_labels > 0] 
    if unique_input_labels.shape[0] == 0: 
        assert np.all(mask == 0)
        return mask

    num_labels = unique_input_labels.shape[0] 
    max_label = np.max(unique_input_labels)
    assert (
            max_label < MAX_INPUT_LABEL_VALUE 
    ), "Input labels out of range for relabeling procedure"
    label_remapper = np.zeros(max_label + 1, np.uint32) 
    label_remapper[unique_input_labels] = np.arange(start_from, num_labels + start_from)

    return label_remapper[mask]


# @process_decorator('GiB')
def overlap_fractions(
        cell_mask,
        nucleus_mask,
        cells_to_nuclei,
        c=False
):
    """Compute the fraction of overlap area of a nucleus and the cell its assigned to.

    This function assumes:
        - `cells_to_nuclei` is a map from cell index to its assigned nucleus,
          which covers more of the cell than any other nucleus
        - `cell_mask` is labled consecutively and the ith label corresponds to
          the ith index in `cells_to_nuclei`.

    Args:
        cell_mask (LabeledMask): The labeled cell mask.
        nucleus_mask (LabeledMask): The labeled nucleus mask.
        cells_to_nuclei (ndarray): A 1D array mapping cells to their assigned
            nucleus.

    Returns:
        ndarray: 1D array containing the fraction of nucleus area that overlaps
            the cell for each cell.
    """
    cell_labels = np.arange(len(cells_to_nuclei))
    nz_assignments = np.nonzero(cells_to_nuclei)
    nz_cell_labels = cell_labels[nz_assignments]
    nz_cells_to_nuclei = cells_to_nuclei[nz_assignments]

    def _max_overlap(val):
        labels, counts = np.unique(val, return_counts=True)
        nonzero = np.nonzero(labels)
        labels = labels[nonzero]
        counts = counts[nonzero]

        if len(counts) == 0:
            return 0

        return np.max(counts)

    if len(nz_cell_labels) == 0:
        return np.zeros(nz_cell_labels.shape, dtype=np.float64)

    # Gives the counts of the label occurring the most times over each cell
    max_counts = ndimage.labeled_comprehension(
        nucleus_mask, cell_mask, nz_cell_labels, _max_overlap, int, 0
    )
    if c:
        areas = ndimage.labeled_comprehension(
            cell_mask, cell_mask, nz_cell_labels, lambda val: val.shape[0], int, 0
        )
    else:
        # Gives the area of each nucleus in pixels
        areas = ndimage.labeled_comprehension(
            nucleus_mask, nucleus_mask, nz_cells_to_nuclei, lambda val: val.shape[0], int, 0
        ) 

    overlap_frac = np.zeros(cells_to_nuclei.shape, dtype=np.float64)
    overlap_frac[nz_assignments] = max_counts / areas 

    return overlap_frac
def instance2semantics(ins):
    """
    instance to semantics
    Args:
        ins(ndarray):labeled instance

    Returns(ndarray):mask
    """
    ins[np.where(ins > 0)] = 1
    return np.array(ins, dtype=np.uint8)

def overlap_v3(secondary_mask_raw, primary_mask_raw, overlap_threshold=0.2, save_path=""):
    secondary_mask_raw = secondary_mask_raw.astype(np.uint8)
    primary_mask_raw = primary_mask_raw.astype(np.uint8)
    secondary_mask = instance2semantics(secondary_mask_raw)
    primary_mask = instance2semantics(primary_mask_raw)
    filtered_secondary_mask = secondary_mask_filter(secondary_mask, primary_mask)
    
    secondary_mask_final = keep_large_nucleus_fragments(secondary_mask, filtered_secondary_mask, threshold= overlap_threshold) #only save pieces larger than threshold

    contours, _ = cv2.findContours(secondary_mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    secondary_boundary = np.zeros_like(secondary_mask_final)
    cv2.drawContours(secondary_boundary, contours, -1, 1, 1)
    primary_mask_add_secondary = np.add(primary_mask, secondary_mask_final)

    save_primary_mask = np.where(secondary_boundary > 0, 0, primary_mask_add_secondary)
    secondary_mask_final = np.where(secondary_boundary > 0, 0, secondary_mask_final)

    return secondary_mask_final, save_primary_mask

def interior_filter(interior_mask: np.ndarray, nuclei_mask: np.ndarray) -> np.ndarray:
    """
    Filter cells by contours, removing interior not overlaped with nuclei
    
    Parameters:
    tissue_mask (numpy.ndarray): Tissue mask image
    cell_mask (numpy.ndarray): Cell mask image
    
    Returns:
    numpy.ndarray: Filtered cell mask image
    """
    contours, _ = cv2.findContours(interior_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        nuclei_roi = nuclei_mask[y:y+h, x:x+w]
        contour_roi = contour - np.array([x, y])

        roi_mask = np.zeros(shape=(h, w), dtype=np.uint8)
        cv2.fillPoly(roi_mask, pts=[contour_roi], color=(1,))

        interior_and_nuclei = cv2.bitwise_and(roi_mask, nuclei_roi)

        total_area = np.sum(roi_mask > 0)
        if total_area > 0:
            overlap_ratio = np.sum(interior_and_nuclei > 0) / total_area
            if overlap_ratio < 0.1:
                cv2.fillPoly(interior_mask, pts=[contour], color=(0,))

    return interior_mask



# @process_decorator('GiB')
def multimodal_merge(
        nuclei_mask_path,
        cell_mask_path,
        interior_mask_path,
        overlap_threshold=0.5,
        save_path="",
        expand_distance=10,
        expand_n_jobs=5,
        final_overlap_threshold=0.1,
):
    """
    assume input instance mask

    overlap between cell mask and interior mask:
    1. overlap == 0, keep both mask
    2. overlap > 0.5, keep cell mask only
    3. 0 < overlap < 0.5, keep cell mask and the non-overlap area of interior mask

    cell mask: cell mask + processed interior mask

    nuclei merge logic:
        1. nuc has less than threshold overlap with cell, save both
        2. nuc has 0 overlap with cell, save both nuc and cell
        3. nuc has more than threshold overlap with cell, save cell only

    Then:
        1. run_fast_correct on output_nuclei_mask
        2. merge expanded nuclei with cell_add_interior again
    """
    import os
    from os.path import join

    nuclei_mask_raw = cbimread(nuclei_mask_path, only_np=True)
    cell_mask_raw = cbimread(cell_mask_path, only_np=True)
    interior_mask_raw = cbimread(interior_mask_path, only_np=True)

    # ----------------------------- merge interior into cell ---------------------------------
    interior_mask_final, cell_add_interior = overlap_v3(
        interior_mask_raw,
        cell_mask_raw,
        overlap_threshold=0.5,
        save_path=save_path
    )

    interior_mask_final = instance2semantics(interior_mask_final)
    nuclei_mask_semantic = instance2semantics(nuclei_mask_raw)
    filter_mask = interior_filter(interior_mask_final, nuclei_mask_semantic)

    '''if save_path != "":
        cbimwrite(
            join(save_path, "cell_add_interior_before_filter.tif"),
            instance2semantics(cell_add_interior) * 255
        )'''

    cell_add_interior = cv2.bitwise_or(cell_mask_raw, filter_mask)

    if save_path != "":
        cbimwrite(join(save_path, "interior_mask_final.tif"), interior_mask_final * 255)
        cbimwrite(
            join(save_path, "cell_mask_add_interior.tif"),
            instance2semantics(cell_add_interior) * 255
        )

    # ----------------------------- first merge nuclei with cell ------------------------------
    output_nuclei_mask, first_merged_mask = overlap_v3(
        nuclei_mask_raw,
        cell_add_interior,
        overlap_threshold=0.8,
        save_path=save_path
    )

    first_merged_mask = instance2semantics(first_merged_mask)

    if save_path != "":
        output_nuclei_path = join(save_path, "output_nuclei_mask.tif")
        cbimwrite(output_nuclei_path, instance2semantics(output_nuclei_mask) * 255)
        cbimwrite(
            join(save_path, "merged_cell_mask.tif"),
            first_merged_mask * 255
        )
    else:
        output_nuclei_path = None

    # ----------------------------- expand nuclei --------------------------------------------
    if output_nuclei_path is not None and os.path.exists(output_nuclei_path):
        fast_mask = run_fast_correct(
            mask_path=output_nuclei_path,
            distance=expand_distance,
            n_jobs=expand_n_jobs
        )
    else:
        fast_mask = output_nuclei_mask

    if save_path != "":
        expand_nuclei_path = join(save_path, "expand_nuclei.tif")
        cbimwrite(expand_nuclei_path, fast_mask)

    # ----------------------------- second merge expanded nuclei with cell --------------------
    secondary_mask_final, final_mask = overlap_v3(
        fast_mask,
        cell_add_interior,
        overlap_threshold=final_overlap_threshold,
        save_path=""
    )

    final_mask = instance2semantics(final_mask).astype(np.uint8)
    #final_mask = break_diagonal_connections(final_mask)

    if save_path != "":
        cbimwrite(join(save_path, "secondary_mask_final.tif"),
                  instance2semantics(secondary_mask_final) * 255)
        cbimwrite(join(save_path, "final_cell_mask.tif"), final_mask * 255)

    return final_mask



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", required=True, help="Output directory for merged results")
    parser.add_argument("--nuc", required=True, help="Path to Nucleus (DAPI) mask file")
    parser.add_argument("--mem", required=True, help="Path to Membrane (CY5) mask file")
    parser.add_argument("--cyto", required=True, help="Path to Interior (TRITC) mask file")
    args = parser.parse_args()
    abs_output_path = os.path.abspath(args.output)
    if not os.path.exists(abs_output_path):
        os.makedirs(abs_output_path, exist_ok=True)
        print(f"Created output directory: {abs_output_path}")

    print(">>> Starting Distributed Mask Fusion")
    print(f"Target Nuclei  (--nuc): {args.nuc}")
    print(f"Target Interior (--cyto): {args.cyto}")
    print(f"Target Membrane  (--mem): {args.mem}")

    save_path = args.output
    nuclei_mask_path = args.nuc
    cell_mask_path = args.mem
    interior_mask_path = args.cyto
    
    multimodal_merge(
        nuclei_mask_path=nuclei_mask_path,
        cell_mask_path=cell_mask_path,
        interior_mask_path=interior_mask_path,
        save_path=save_path
    )

    print(f"\n[Success] Final Merged Mask: {os.path.join(abs_output_path, 'merged_cell_mask.tif')}")