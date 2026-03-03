import numpy as np
from skimage import measure, io, color
from collections import defaultdict
import os
import argparse
from skimage import io, measure, color
import gzip


def preprocess_mask(mask):

    binary_mask = (mask > 0).astype(np.uint8)
    nonzero_coords = np.argwhere(binary_mask)
    
    if len(nonzero_coords) == 0:
        print("Warning: No tissue region found in mask")
        return mask, (0, mask.shape[0], 0, mask.shape[1])
    
    min_row = np.min(nonzero_coords[:, 0])
    max_row = np.max(nonzero_coords[:, 0])
    min_col = np.min(nonzero_coords[:, 1])
    max_col = np.max(nonzero_coords[:, 1])
    
    margin = 5
    height, width = binary_mask.shape
    min_row = max(0, min_row - margin)
    max_row = min(height, max_row + margin)
    min_col = max(0, min_col - margin)
    max_col = min(width, max_col + margin)
    
    cropped_mask = binary_mask[min_row:max_row, min_col:max_col]
    bbox = (min_row, max_row, min_col, max_col)

    return cropped_mask, bbox

def scan_based_relabeling(labels, scan_scale: float = 1.5):

    props = measure.regionprops(labels)
    
    if len(props) == 0:
        return labels.copy(), []
    
    areas = [prop.area for prop in props]
    avg_area = np.mean(areas)
    step_size = max(500, int(np.sqrt(avg_area) * scan_scale))
    
    height, width = labels.shape
    
    centroids = np.array([prop.centroid for prop in props])
    labels_list = [prop.label for prop in props]
    
    scan_regions = defaultdict(list)
    region_to_scan_index = {}
    
    scan_index = 0
    
    y_starts = range(0, height, step_size)
    x_starts = range(0, width, step_size)
    
    for start_y in y_starts:
        for start_x in x_starts:
            end_y = min(start_y + step_size, height)
            end_x = min(start_x + step_size, width)
            
            in_window_mask = (
                (centroids[:, 0] >= start_y) & (centroids[:, 0] < end_y) &
                (centroids[:, 1] >= start_x) & (centroids[:, 1] < end_x)
            )
            
            unassigned_mask = np.array([label not in region_to_scan_index for label in labels_list])
            valid_indices = np.where(in_window_mask & unassigned_mask)[0]
            
            if len(valid_indices) > 0:
                scan_index += 1
                window_regions = [props[i] for i in valid_indices]
                scan_regions[scan_index] = window_regions
                for i in valid_indices:
                    region_to_scan_index[labels_list[i]] = scan_index
    
    unassigned_indices = [i for i, label in enumerate(labels_list) if label not in region_to_scan_index]
    
    if unassigned_indices:
        scan_centers = {}
        for scan_idx, regions in scan_regions.items():
            if regions:
                region_centroids = np.array([r.centroid for r in regions])
                scan_centers[scan_idx] = np.mean(region_centroids, axis=0)
        
        for idx in unassigned_indices:
            prop = props[idx]
            y, x = centroids[idx]
            
            distances = []
            scan_indices = []
            for scan_idx, center in scan_centers.items():
                distance = np.sqrt((y - center[0])**2 + (x - center[1])**2)
                distances.append(distance)
                scan_indices.append(scan_idx)
            
            if distances:
                closest_idx = np.argmin(distances)
                closest_scan_index = scan_indices[closest_idx]
                scan_regions[closest_scan_index].append(prop)
                region_to_scan_index[prop.label] = closest_scan_index
    
    scan_info = []
    for scan_index_n, regions in scan_regions.items():
        if not regions:
            continue
            
        region_centroids = np.array([prop.centroid for prop in regions])
        region_labels = [prop.label for prop in regions]
        region_areas = [prop.area for prop in regions]
        
        sort_indices = np.lexsort((region_centroids[:, 1], region_centroids[:, 0]))
        
        for local_index_m, orig_idx in enumerate(sort_indices, 1):
            prop = regions[orig_idx]
            scan_info.append({
                'old_label': region_labels[orig_idx],
                'scan_index_n': scan_index_n,
                'local_index_m': local_index_m,
                'centroid': region_centroids[orig_idx],
                'area': region_areas[orig_idx],
                'index_2d': (scan_index_n, local_index_m)
            })
    
    scan_info_sorted = sorted(scan_info, key=lambda x: (x['scan_index_n'], x['local_index_m']))
    
    label_mapping = {}
    for final_label_id, info in enumerate(scan_info_sorted, 1):
        label_mapping[info['old_label']] = final_label_id
        info['final_label_id'] = final_label_id
    
    new_labels = np.zeros_like(labels)
    
    old_labels_flat = labels.ravel()
    new_labels_flat = np.zeros_like(old_labels_flat)
    
    for old_label, new_label in label_mapping.items():
        mask = old_labels_flat == old_label
        new_labels_flat[mask] = new_label
    
    new_labels = new_labels_flat.reshape(labels.shape)
    
    return new_labels

def restore_labels_to_original(new_labels, bbox, original_shape):

    min_row, max_row, min_col, max_col = bbox
    cropped_height, cropped_width = new_labels.shape
    original_height, original_width = original_shape
    
    restored_labels = np.zeros(original_shape, dtype=new_labels.dtype)
    
    restored_labels[min_row:min_row + cropped_height, min_col:min_col + cropped_width] = new_labels
    
    return restored_labels

def export_foreground_coords(mask_path: str, output_dir: str, min_size: int = -1, scan_scale: float = 1.5):
    group_name = "Group 01"
    bin_size = 1

    mask = io.imread(mask_path)
    if mask.ndim > 2:
        mask = color.rgb2gray(mask)

    mask = (mask > 0).astype(np.uint8)
    if min_size != -1:
        from skimage.morphology import remove_small_objects
        mask = remove_small_objects(mask>0, min_size=min_size, connectivity=2)
    
    original_shape = mask.shape
    
    processed_mask, bbox = preprocess_mask(mask)
    
    labels = measure.label(processed_mask, connectivity=2).astype(np.uint16)
    print(f"labels: {np.unique(labels)}\n{len(np.unique(labels))} labels found with min-size:{min_size} filtered.")

    new_labels = scan_based_relabeling(labels, scan_scale)
    scan_labels = restore_labels_to_original(new_labels, bbox, original_shape)

    ys, xs = np.nonzero(scan_labels)
    lbl_vals = scan_labels[ys, xs]
    order = np.argsort(lbl_vals)
    ys, xs, lbl_vals = ys[order], xs[order], lbl_vals[order]

    buffer_size = 1_000_000
    buffer = []

    output_csv = os.path.join(output_dir, 'split_labels_bin1.csv.gz')

    with gzip.open(output_csv, "wt", encoding="utf-8", newline="") as f:
        f.write("X coordinate,Y coordinate,Group name,Label name,Bin size\n")
        for i, (x, y, lbl) in enumerate(zip(xs, ys, lbl_vals)):
            buffer.append(f"{x},{y},{group_name},Label {lbl:02d},{bin_size}\n")
            if len(buffer) >= buffer_size:
                f.writelines(buffer)
                buffer.clear()
        if buffer:
            f.writelines(buffer)


def parse_args():
    p = argparse.ArgumentParser(description="Export foreground label pixel coordinates to CSV.GZ")
    p.add_argument("--mask_path", "-m", required=True, type=str, help="input mask path")
    p.add_argument("--output_dir", "-o", required=True, type=str, help="output file path")
    p.add_argument("--min_size", "-s", required=False, type=int, help="remove small objects", default=-1)
    p.add_argument("--scan_scale", "-sc", required=False, type=float, help="control scan step size scale", default=1.5)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    export_foreground_coords(mask_path=args.mask_path, output_dir=args.output_dir)