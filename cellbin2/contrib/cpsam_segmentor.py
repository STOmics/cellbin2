import os
import argparse
from tqdm import tqdm, trange
import pip
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import tifffile
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
def run_cpsam(model_path, input_path, output_dir, batch_size=8, use_gpu=False):

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading model from: {model_path}")
    model = models.CellposeModel(
        gpu=use_gpu,
        pretrained_model=model_path
    )

    if os.path.isfile(input_path):
        test_files = [input_path]
    elif os.path.isdir(input_path):
        test_files = [os.path.join(input_path, f) for f in os.listdir(input_path)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    else:
        raise ValueError(f"Invalid input path: {input_path}")

    if not test_files:
        raise ValueError(f"No valid image files found in: {input_path}")

    print(f"Processing {len(test_files)} image(s)...")
    for file in tqdm(test_files):
        try:
            image = io.imread(file)
            masks = model.eval(image, batch_size=batch_size)[0]
            base_name = os.path.splitext(os.path.basename(file))[0]
            save_path = os.path.join(output_dir, f"{base_name}_cpsam_masks.tif")
            io.imsave(save_path, masks)
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")

    print("Inference completed.")



def instance2semantics(ins):
    """
    instance to semantics
    Args:
        ins(ndarray):labeled instance

    Returns(ndarray):mask
    """
    ins[np.where(ins > 0)] = 1
    return np.array(ins, dtype=np.uint8)

def predict_cpsam(image_path, batch_size=8, use_gpu=True):
    """
    use cellpose_sam to generate single mask
    Args:
        model_path (str): model weights path
        image (np.ndarray): input image to numpy array
        batch_size (int): batch size
        use_gpu (bool): use gpu or not
    Returns:
        masks (np.ndarray): segmentation masks
    """
    try:
        import cellpose
    except ImportError:
        pip.main(['install', 'git+https://www.github.com/mouseland/cellpose.git'])
    if cellpose.version != '4.0.7':
        pip.main(['install', 'git+https://www.github.com/mouseland/cellpose.git'])
    import cellpose

    from cellpose import models, core, io, plot

    image = io.imread(str(image_path))
    model = models.CellposeModel(gpu=use_gpu)
    flow_threshold = 0.4
    cellprob_threshold = 0.0
    tile_norm_blocksize = 0
    masks, flows, styles = model.eval(
            image,
            batch_size=32,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            normalize={"tile_norm_blocksize": tile_norm_blocksize}
        )

        # transfer to semantic segmentation
    semantics = instance2semantics(masks)
    #semantics[semantics > 0] = 255
    return semantics



path = "/storeData/USER/data/01.CellBin/00.user/wangaoli/data/result/时空多蛋白数据/fake_chip_run/Y00038H1/Y00038H1_CY5_IF_stitch.tif"
mask = predict_cpsam(path)
tifffile.imwrite("/storeData/USER/data/01.CellBin/00.user/wangaoli/data/result/时空多蛋白数据/fake_chip_run/20X_mid_cut_s/cellpose4_cy5.tif",mask)