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
from cellbin2.contrib.cell_filter import FilterCells, filter_pipline
from cellbin2.utils.cell_shape import f_main as cell_shape_f_main


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
        from cellbin2.contrib.alignment.basic import TemplateInfo
        cm._template = TemplateInfo(template_recall=1.,
                                    template_valid_area=1.,
                                    trackcross_qc_pass_flag=1,
                                    trackline_channel=0,
                                    rotation=0.,
                                    scale_x=1., scale_y=1.,
                                    template_points=track_points)

        # _gene_mat
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

def delete_doublet_from_expanded_mask(filter_txt, mask_path, out_path):
    import cv2
    import pandas as pd
    import numpy as np
    import tifffile
    from cellbin2.image import cbimwrite

    df = pd.read_csv(filter_txt, sep="\t")
    df = df[df["delete_flag"] == 1].copy()

    img = tifffile.imread(str(mask_path))

    if len(df) == 0:
        clog.info("No doublet cells found, copy expanded mask directly.")
        cbimwrite(str(out_path), img)
        return str(out_path)

    # x/y columns are always written by FilterCells.result_to_txt.
    if {"x", "y"}.issubset(set(df.columns)):
        coords = []
        for row in df.itertuples(index=False):
            coords.append((int(getattr(row, "x")), int(getattr(row, "y"))))
    else:
        raise ValueError(
            "filter.txt has no x/y columns. "
            "Cannot delete doublet cells from expanded mask."
        )

    mask01 = (img > 0).astype(np.uint8)

    h, w = mask01.shape

    removed_components = 0
    skipped_background = 0
    skipped_outside = 0

    for x, y in coords:
        if not (0 <= x < w and 0 <= y < h):
            skipped_outside += 1
            continue

        if mask01[y, x] == 0:
            skipped_background += 1
            continue

        num, _, _, _ = cv2.floodFill(
            mask01,
            None,
            seedPoint=(x, y),
            newVal=0,
            loDiff=0,
            upDiff=0,
            flags=4,
        )

        if num > 0:
            removed_components += 1

    out = img.copy()
    out[mask01 == 0] = 0

    cbimwrite(str(out_path), out)
    return str(out_path)

def extract4matrix(
        p_naming: naming.DumpPipelineFileNaming,
        image_file: ProcFile,
        m_naming: naming.DumpMatrixFileNaming,
        doublet_filter: bool = False,
):
    """
    Extracts matrix data for stitched images based on cell and tissue masks.

    Args:
        p_naming (naming.DumpPipelineFileNaming): Naming convention for pipeline files.
        image_file (ProcFile): Processed image file.
        m_naming (naming.DumpMatrixFileNaming): Naming convention for matrix files.
    """
    # Check if tissue mask is considered for cell matrix extraction, needs confirmation
    import os
    from cellbin2.matrix.matrix import save_cell_bin_data, save_tissue_bin_data, cMatrix
    cell_mask_path = str(p_naming.final_nuclear_mask)
    tissue_mask_path = p_naming.final_tissue_mask
    cell_correct_mask_path =str(p_naming.final_cell_mask)
    c_inp = None
    path = os.path.dirname(cell_mask_path)

    clog.info(f"Matrix input, use save_cell_bin_data: {image_file.file_path}")

    binx = 1

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
        cgeffile = str(m_naming.cell_correct_bin_matrix)

        result = save_cell_bin_data(
                image_file.file_path,
                cgeffile,
                cell_correct_mask_path,
            )
        cell_shape_f_main(cgeffile)

        final_cellbin_gef_for_stereo = cgeffile
    else:
        clog.info(f"{cell_correct_mask_path} not exists, skip unfiltered expanded cellbin gef generation")
        final_cellbin_gef_for_stereo = None

    if doublet_filter:
        raw_cell_gef = str(m_naming.cell_bin_matrix)
        filter_txt = os.path.join(path, f"{m_naming.sn}.filter.txt")
        filter_gef = os.path.join(path, f"{m_naming.sn}.filter.gef")

        filtered_cell_correct_mask_path = os.path.join(
            path,
            f"{m_naming.sn}_final_cell_mask_after_doublet_delete.tif"
        )

        filtered_cell_correct_gef = Path(path) / f"{m_naming.sn}.filtered.cellbin.gef"

        if (
            Path(raw_cell_gef).exists()
            and Path(cell_mask_path).exists()
            and Path(cell_correct_mask_path).exists()
        ):
            filter_pipline(
                raw_cell_gef,
                cell_mask_path,
                output_filter_file=filter_txt,
                output_filter_gef=filter_gef,
            )

            filtered_cell_correct_mask_path = delete_doublet_from_expanded_mask(
                filter_txt=filter_txt,
                mask_path=cell_correct_mask_path,
                out_path=filtered_cell_correct_mask_path,
            )

            if Path(filtered_cell_correct_mask_path).exists():
                result = save_cell_bin_data(
                        image_file.file_path,
                        str(filtered_cell_correct_gef),
                        filtered_cell_correct_mask_path,
                    )
                cell_shape_f_main(str(filtered_cell_correct_gef))

                final_cellbin_gef_for_stereo = filtered_cell_correct_gef

                clog.info("[doublet filter output]")
                clog.info(f"  unfiltered expanded cellbin gef = {cgeffile}")
                clog.info(f"  filtered expanded mask          = {filtered_cell_correct_mask_path}")
                clog.info(f"  filtered expanded cellbin gef   = {filtered_cell_correct_gef}")
            else:
                clog.warning(
                    f"filtered expanded mask not exists: {filtered_cell_correct_mask_path}. "
                    "Skip filtered cellbin gef generation."
                )
        else:
            clog.warning(
                "doublet_filter=True, but raw cgef/raw mask/expanded mask missing. "
                "Skip doublet filter."
            )

    if image_file.tech == TechType.Transcriptomics and final_cellbin_gef_for_stereo is not None:
        generate_stereo_file(
            save_path=p_naming.stereo,
            cellbin_gef=Path(final_cellbin_gef_for_stereo)
        )


def main():
    pass


if __name__ == '__main__':
    main()
