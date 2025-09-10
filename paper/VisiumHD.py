import pandas as pd
from tqdm import tqdm
import os
import gzip
import scanpy as sc
import numpy as np
from PIL import Image, ImageFile
from tifffile import TiffFile
import os

Image.MAX_IMAGE_PIXELS = None  
ImageFile.LOAD_TRUNCATED_IMAGES = True  


def read_large_image(image_path):
    """
    读取大图像（支持 .tif 和 .btf 格式）。
    如果是 .btf 文件，将其转换为 .tif 文件后读取。
    """
    # 检查文件扩展名
    ext = os.path.splitext(image_path)[1].lower()

    if ext == ".btf":
        # 转换 .btf 为 .tif
        output_file = os.path.splitext(image_path)[0] + ".tif"
        try:
            with Image.open(image_path) as img:
                img.save(output_file, format="TIFF")
            print(f"成功将 .btf 文件转换为 .tif：{output_file}")
            image_path = output_file  # 更新路径为转换后的 .tif 文件
        except Exception as e:
            raise ValueError(f".btf 文件转换失败：{e}")

    # 读取 .tif 文件
    try:
        with TiffFile(image_path) as tif:
            image = tif.asarray()
        print(f"成功读取图像：{image_path}")
        return image
    except Exception as e:
        raise ValueError(f"图像读取失败：{e}")
def crop_image_by_coords(image, spatial_coords):
    """
    根据空间坐标裁剪图像，并更新裁剪后的坐标。
    :param image: numpy.ndarray, 原始图像
    :param spatial_coords: numpy.ndarray, 空间坐标，形状为 (n, 2)，列为 [列坐标, 行坐标]
    :return: 裁剪后的图像，更新的空间坐标
    """
    # 计算行列值的最大最小值
    min_row, max_row = np.min(spatial_coords[:, 1]), np.max(spatial_coords[:, 1])
    min_col, max_col = np.min(spatial_coords[:, 0]), np.max(spatial_coords[:, 0])

    # 裁剪图像
    cropped_image = image[min_row:max_row + 1, min_col:max_col + 1]

    # 更新坐标
    updated_coords = spatial_coords - [min_col, min_row]

    return cropped_image, updated_coords

def main(args, para):
    input_image = args.input_image
    parquet_file = args.parquet_file
    h5_file = args.h5_file
    output_path = args.output_path

    # 读取10X数据
    adata = sc.read_10x_h5(h5_file)
    # 读取空间信息
    spatial_data = pd.read_parquet(parquet_file)
    # 读取图片
    image = read_large_image(input_image)
    
    # 索引对齐，确保空间信息和表达矩阵的条目一致
    spatial_data = spatial_data.set_index('barcode')
    spatial_data = spatial_data.loc[adata.obs_names]

    # 将空间信息与表达矩阵结合
    adata.obs = adata.obs.join(spatial_data, how='inner')

    # 过滤掉负数的行（在过滤前已经结合了数据）
    adata = adata[
        (adata.obs['in_tissue'] == True) &
        (adata.obs['pxl_row_in_fullres'] >= 0) &
        (adata.obs['pxl_col_in_fullres'] >= 0)
    ]

    # 提取空间坐标并四舍五入为整数
    spatial_coords = adata.obs[['pxl_col_in_fullres', 'pxl_row_in_fullres']].values
    spatial_coords = np.round(spatial_coords).astype(int)  # 四舍五入为整数

    # 裁剪图像并更新坐标
    cropped_image, updated_coords = crop_image_by_coords(image, spatial_coords)

    # 保存裁剪后的图像
    cropped_image_path = os.path.join(output_path, "A04547A1C3.tif")
    Image.fromarray(cropped_image).save(cropped_image_path)
    print(f"裁剪后的图像已保存：{cropped_image_path}")

    # 更新坐标到 adata.obs
    adata.obs['pxl_col_in_cropped'] = updated_coords[:, 0]
    adata.obs['pxl_row_in_cropped'] = updated_coords[:, 1]

    # 保存更新后的空间信息为 Parquet 格式
    updated_coords_path = os.path.join(output_path, "tissue_positions.parquet")
    adata.obs.to_parquet(updated_coords_path, index=True)
    print(f"更新后的坐标已保存为 Parquet 文件：{updated_coords_path}")

    # 获取基因ID
    gene_ids = adata.var_names.tolist()

    # 获取稀疏矩阵中非零元素的索引和表达量
    non_zero_indices = adata.X.nonzero()  # 获取非零元素的索引
    cell_indices = non_zero_indices[0]    # 细胞索引
    gene_indices = non_zero_indices[1]    # 基因索引
    expression_values = adata.X[cell_indices, gene_indices].A1  # 获取对应的表达量

    # 获取相应的空间坐标
    cell_spatial_coords = updated_coords[cell_indices]

    # 获取基因ID
    genes = [gene_ids[i] for i in gene_indices]

    # 定义批次大小
    batch_size = 50000  # 每个批次处理的基因数量
    num_batches = len(genes) // batch_size + 1

    # 创建输出文件路径
    # 动态生成输出文件路径
    input_filename = os.path.splitext(os.path.basename(input_image))[0]  # 提取输入图像文件名（不带扩展名）
    output_file = os.path.join(output_path, "A04547A1C3.gem.gz")  # 拼接为输出路径+文件名.gem.gz

    # 删除已有的文件，确保重新生成
    if os.path.exists(output_file):
        os.remove(output_file)

    # 列名定义，确保一致性
    gem_columns = ['geneID', 'x', 'y', 'MIDCount']

    # 分批处理数据
    for batch_num in tqdm(range(num_batches), desc="Processing batches"):
        # 计算当前批次的起始和结束索引
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, len(genes))

        # 获取当前批次的数据
        batch_genes = genes[start_idx:end_idx]
        batch_spatial_coords = cell_spatial_coords[start_idx:end_idx]
        batch_expression_values = expression_values[start_idx:end_idx]

        # 创建当前批次的 DataFrame
        batch_df = pd.DataFrame({
            'geneID': batch_genes,
            'x': batch_spatial_coords[:, 0],  # 空间坐标 x
            'y': batch_spatial_coords[:, 1],  # 空间坐标 y
            'MIDCount': batch_expression_values,   # 基因的表达量
        })

        # 确保列名干净且一致
        batch_df.columns = batch_df.columns.str.strip().str.replace('\r', '', regex=False)

        # 如果是第一个批次，在文件开头添加特定的内容
        if batch_num == 0:
            # 打开文件并写入自定义的头部内容（不在压缩内容前直接写入）
            with gzip.open(output_file, 'wt', encoding='utf-8') as f:
                f.write('# x_start=0\n')
                f.write('# y_start=0\n')
                # 将列名写入文件
                batch_df.to_csv(f, sep='\t', header=True, index=False)

        # 确保列的顺序正确
        batch_df = batch_df[gem_columns]

        # 将当前批次数据追加到 GEM 文件（以压缩格式写入）
        with gzip.open(output_file, 'at', encoding='utf-8') as f:
            batch_df.to_csv(f, sep='\t', header=False, index=False)


if __name__ == '__main__':  # main()
    import argparse

    _VERSION_ = '0.1'

    parser = argparse.ArgumentParser(
        description="This is CellBin V2.0 Pipeline of visium HD",
    )
    parser.add_argument("-i", "--input_image", action="store", type=str,
                        help=f"The path of input file.")
    parser.add_argument("-h5", "--h5_file", action="store", type=str,
                        help="The path of h5 file.")
    parser.add_argument("-pa", "--parquet_file", action="store", type=str,
                        help="The path of parquet file.")
    parser.add_argument("-o", "--output_path", action="store", type=str, required=True,
                        help="The results output folder.")

    parser.set_defaults(func=main)
    (para, args) = parser.parse_known_args()
    para.func(para, args)
