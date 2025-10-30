import argparse
import numpy as np
from skimage import io, measure, color
import numpy as np
import gzip
import os

def export_foreground_coords(mask_path: str, output_csv: str, min_size: int = -1):
    group_name = "Group 01"
    bin_size = 1

    mask = io.imread(mask_path)
    if mask.ndim > 2:
        mask = color.rgb2gray(mask)
    mask = (mask > 0).astype(np.uint8)

    if min_size != -1:
        from skimage.morphology import remove_small_objects
        mask = remove_small_objects(mask>0, min_size=min_size, connectivity=2)

    labels = measure.label(mask, connectivity=2).astype(np.uint16)
    print(f"labels: {np.unique(labels)}\n{len(np.unique(labels))} labels found with min-size:{min_size} filtered.")

    ys, xs = np.nonzero(labels)
    lbl_vals = labels[ys, xs]
    order = np.argsort(lbl_vals)
    ys, xs, lbl_vals = ys[order], xs[order], lbl_vals[order]

    buffer_size = 1_000_000
    buffer = []
    if output_csv.split('.')[-1] != '.csv':
        output_csv = os.path.join(output_csv, 'split_labels_bin1.csv')

    # with open(output_csv, "w", encoding="utf-8", newline="") as f:
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
    p = argparse.ArgumentParser(description="Export foreground label pixel coordinates to CSV")
    p.add_argument("--mask", "-m", required=True, type=str, help="input mask path")
    p.add_argument("--out", "-o", required=True, type=str, help="output file path")
    p.add_argument("--min_size", "-s", required=True, type=int, help="remove small objects")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    export_foreground_coords(args.mask, args.out, min_size=50000)


