from os.path import join

import pandas as pd
import gzip
import tifffile


def xenium_csv_to_gem(xenium_csv_file_path, gem_save_file_path):
    """
    xenium_csv_file_path: transcripts.csv.gz
    """
    with gzip.open(xenium_csv_file_path, 'rt') as f:
        df = pd.read_csv(f)
        df.columns = df.columns.str.strip()

    # 先除以 0.2125统一坐标，再四舍五入为整数
    # reference: https://cf.10xgenomics.com/supp/xenium/xenium_documentation.html
    df['x_location'] = (df['x_location'] / 0.2125).round().astype(int)
    df['y_location'] = (df['y_location'] / 0.2125).round().astype(int)

    print("x range:", df['x_location'].min(), df['x_location'].max())
    print("y range:", df['y_location'].min(), df['y_location'].max())

    gem_data = df[['feature_name', 'x_location', 'y_location']].copy()
    gem_data['MIDCount'] = 1

    with open(gem_save_file_path, 'w', newline='\n') as f:
        f.write("# x_start=0\n")
        f.write("# y_start=0\n")
        f.write("geneID\tx\ty\tMIDCount\n")

        for _, row in gem_data.iterrows():
            f.write(f"{row['feature_name']}\t{row['x_location']}\t{row['y_location']}\t{row['MIDCount']}\n")

    print(f"GEM file has been written to {gem_save_file_path}")


def extract_ome_file(ome_file_path, save_path):
    """
    read multi-file OME-TIFF format
    reference: https://www.10xgenomics.com/support/software/xenium-onboard-analysis/latest/advanced/example-code#view-imgs
    ome_file_path: "morphology_focus/morphology_focus_0000.ome.tif"

    - morphology_focus_0000.ome.tif: DAPI image
    - morphology_focus_0001.ome.tif: boundary (ATP1A1/E-Cadherin/CD45) image
    - morphology_focus_0002.ome.tif: interior - RNA (18S) image
    - morphology_focus_0003.ome.tif: interior - protein (alphaSMA/Vimentin) image
    reference: https://www.10xgenomics.com/support/software/xenium-onboard-analysis/latest/analysis/xoa-output-understanding-outputs#images
    """
    fullres_multich_img = tifffile.imread(ome_file_path, is_ome=True, level=0, aszarr=False)
    print(fullres_multich_img.shape)
    print(fullres_multich_img.shape[0])
    from cellbin2.image import cbimwrite
    cbimwrite(join(save_path, f"morphology_focus_DAPI.tif"), fullres_multich_img[0, ...])
    cbimwrite(join(save_path, f"morphology_focus_boundary.tif"), fullres_multich_img[1, ...])
    cbimwrite(join(save_path, f"morphology_focus_interior_rna.tif"), fullres_multich_img[2, ...])
    cbimwrite(join(save_path, f"morphology_focus_interior_protein.tif"), fullres_multich_img[3, ...])


def main():
    # step 1: convert Xenium data to cellbinv2 format
    # images
    # ome_file_path = "/media/Data1/user/dengzhonghan/data/cellbin2paper/Xenium_V1_humanLung_Cancer_FFPE/morphology_focus_0000.ome.tif"
    # save_path = "/media/Data1/user/dengzhonghan/data/cellbin2paper/Xenium_V1_humanLung_Cancer_FFPE/cellbinv2_input"
    # extract_ome_file(ome_file_path=ome_file_path, save_path=save_path)

    # transcript
    xenium_csv_path = "/media/Data1/user/dengzhonghan/data/cellbin2paper/Xenium_V1_humanLung_Cancer_FFPE/transcripts.csv.gz"
    gem_save_file_path = \
        "/media/Data1/user/dengzhonghan/data/cellbin2paper/Xenium_V1_humanLung_Cancer_FFPE/cellbinv2_input/transcripts.gem"
    xenium_csv_to_gem(xenium_csv_path, gem_save_file_path)


if __name__ == '__main__':
    df = pd.read_csv(
        "/media/Data1/user/dengzhonghan/data/cellbin2paper/Xenium_V1_humanLung_Cancer_FFPE/cellbinv2_input/transcripts.gem",
    sep="\t",
    skiprows=2)
    df2 = pd.read_csv(
        "/media/Data1/user/dengzhonghan/data/cellbin2paper/Xenium_V1_humanLung_Cancer_FFPE/cellbinv2_input/B03205D314.gem",
        skiprows=2,sep="\t",

    )
    # header=0)
    print()
