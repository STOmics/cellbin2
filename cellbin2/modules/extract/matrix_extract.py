from pathlib import Path

import numpy as np

from cellbin2.image import cbimwrite
from cellbin2.modules.metadata import ProcFile
from cellbin2.modules import naming
from cellbin2.matrix.matrix import cMatrix
from cellbin2.utils.stereo_chip import StereoChip
from cellbin2.utils.config import Config
from cellbin2.utils import clog
from cellbin2.utils.stereo import generate_stereo_file
from cellbin2.utils.common import TechType
from cellbin2.image.augmentation import f_resize
from cellbin2.utils.stereo_chip import StereoChip
from cellbin2.image.augmentation import f_ij_16_to_8


def extract4stitched(
        image_file: ProcFile,
        param_chip: StereoChip,
        m_naming: naming.DumpMatrixFileNaming,
        config: Config,
        detect_feature: bool = True,
):
    """
    Extracts matrix data for a stitched image.

    Args:
        image_file (ProcFile): The image file to process.
        param_chip (StereoChip): Parameters for the stereo chip.
        m_naming (naming.DumpMatrixFileNaming): Naming convention for matrix files.
        config (Config): Configuration settings.

    """
    cm = cMatrix()
    cm.read(file_path=Path(image_file.file_path))
    binx = cm.binx

    cm.check_standards(config.genetic_standards)

    if binx != 1:
        sc = StereoChip()
        sc.parse_info(chip_no=m_naming.sn)
        track_points = sc.template_points
        # track_points = np.loadtxt('/media/Data1/user/hedongdong/wqs/00.code/03.CellBin2/bin_X/data/SS200000135TL_D1_Transcriptomics_matrix_template.txt')
        # print(track_points.shape)
        # np.savetxt('/media/Data1/user/hedongdong/wqs/00.code/03.CellBin2/bin_X/data/track_points.txt', track_points)
        from cellbin2.contrib.alignment.basic import TemplateInfo
        cm._template = TemplateInfo(template_recall=1.,
                                    template_valid_area=1.,
                                    trackcross_qc_pass_flag=1,
                                    trackline_channel=0,
                                    rotation=0.,
                                    scale_x=1., scale_y=1.,
                                    template_points=track_points)

        # 对_gene_mat插值放大
        cbimwrite(str(m_naming.heatmap).replace('.tif', f'_bin{binx}.tif'), cm.heatmap)
        gene_mat = cm._gene_mat
        gene_mat = f_ij_16_to_8(gene_mat)
        if gene_mat is not None and gene_mat.size > 0:
            new_shape = (gene_mat.shape[0] * binx, gene_mat.shape[1] * binx)
            gene_mat_resized = f_resize(gene_mat, shape=new_shape, mode="BICUBIC")
            cm._gene_mat = gene_mat_resized

    elif detect_feature:
        cm.detect_feature(ref=param_chip.fov_template,
                          chip_size=min(param_chip.chip_specif))
        gene_tps = cm.template.template_points[:, :2]  # StereoMap is only compatible with n×2
        np.savetxt(m_naming.matrix_template, gene_tps)
    cbimwrite(m_naming.heatmap, cm.heatmap)
    return cm, binx


def extract4matrix(
        p_naming: naming.DumpPipelineFileNaming,
        image_file: ProcFile,
        m_naming: naming.DumpMatrixFileNaming,
):
    """
    Extracts matrix data for stitched images based on cell and tissue masks.

    Args:
        p_naming (naming.DumpPipelineFileNaming): Naming convention for pipeline files.
        image_file (ProcFile): Processed image file.
        m_naming (naming.DumpMatrixFileNaming): Naming convention for matrix files.
    """
    # Check if tissue mask is considered for cell matrix extraction, needs confirmation
    from cellbin2.matrix.matrix import save_cell_bin_data, save_tissue_bin_data, cMatrix
    cell_mask_path = p_naming.final_nuclear_mask
    tissue_mask_path = p_naming.final_tissue_mask
    cell_correct_mask_path = p_naming.final_cell_mask
    c_inp = None

    binx, _, _ = cMatrix.gef_gef_shape(image_file.file_path)

    if Path(tissue_mask_path).exists():
        save_tissue_bin_data(
            image_file.file_path,
            str(m_naming.tissue_bin_matrix),
            tissue_mask_path,
            bin_siz=binx
        )
        c_inp = m_naming.tissue_bin_matrix
        if image_file.tech == TechType.Transcriptomics:
            generate_stereo_file(
                save_path=p_naming.stereo,
                gef=m_naming.tissue_bin_matrix
            )
    else:
        clog.info(f"{tissue_mask_path} not exists, skip tissue gef generation")
    if c_inp is None:
        c_inp = image_file.file_path
    if Path(cell_mask_path).exists():
        save_cell_bin_data(
            c_inp,
            str(m_naming.cell_bin_matrix),
            cell_mask_path)
    else:
        clog.info(f"{cell_mask_path} not exists, skip nuclear gef generation")
    if Path(cell_correct_mask_path).exists():
        save_cell_bin_data(
            c_inp,
            str(m_naming.cell_correct_bin_matrix),
            cell_correct_mask_path
        )
        if image_file.tech == TechType.Transcriptomics:
            generate_stereo_file(
                save_path=p_naming.stereo,
                cellbin_gef=m_naming.cell_correct_bin_matrix
            )
    else:
        clog.info(f"{cell_mask_path} not exists, skip cellbin gef generation")


def main():
    pass


if __name__ == '__main__':
    main()
