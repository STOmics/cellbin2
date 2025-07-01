import os
import argparse
from cellpose import models, io
from tqdm import tqdm
import numpy as np

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

def predict_cpsam(model_path, image, batch_size=8, use_gpu=False):
    """
    use cellpose_sam to generate single mask
    Args:
        model_path (str): 模型权重路径
        image (np.ndarray): 输入图片（已读取为numpy数组）
        batch_size (int): batch size
        use_gpu (bool): 是否用GPU
    Returns:
        masks (np.ndarray): 分割mask
    """
    model = models.CellposeModel(
        gpu=use_gpu,
        pretrained_model=model_path
    )
    masks = model.eval(image, batch_size=batch_size)[0]
    return masks

def main():
    parser = argparse.ArgumentParser(description='Cellpose-SAM fine-tuned model inference')
    parser.add_argument('-p', '--model_path', type=str, required=True,
                        help='Path to the fine-tuned model (e.g., /path/to/finetuned_model)')
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Path to a single image file or a directory containing images')
    parser.add_argument('-o', '--output_dir', type=str, required=True,
                        help='Directory to save output masks')
    parser.add_argument('-b', '--batch_size', type=int, default=8,
                        help='Batch size (default: 8)')
    parser.add_argument('-g', '--use_gpu', action='store_true',
                        help='Use GPU if available')
    args = parser.parse_args()

    run_cpsam(
        model_path=args.model_path,
        input_path=args.input,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        use_gpu=args.use_gpu
    )


if __name__ == "__main__":
    main()
