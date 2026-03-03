from cellbin2.modules.naming import DumpMatrixFileNaming, DumpPipelineFileNaming, DumpImageFileNaming
from pathlib import Path
from typing import Union
import enum


class MatrixRunState(object):
    def __init__(self, src_file: Union[str, Path], dst_dir: Union[str, Path]):
        self._naming = DumpMatrixFileNaming(src_file)
        if type(dst_dir) is str: dst_dir = Path(dst_dir)
        self._dst_dir = dst_dir

    @property
    def cell_bin_matrix(self, ):
        return (self._dst_dir / self._naming.cell_bin_matrix).exists()

    @property
    def cell_correct_bin_matrix(self, ):
        return (self._dst_dir / self._naming.cell_correct_bin_matrix).exists()

    @property
    def tissue_bin_matrix(self, ):
        return (self._dst_dir / self._naming.tissue_bin_matrix).exists()

    @property
    def matrix_template(self, ):  # matrix template file
        return (self._dst_dir / self._naming.matrix_template).exists()


class ImageRunState(object):
    def __init__(self, dst_dir: Union[str, Path], sn, stain_type):
        self._naming = DumpImageFileNaming(sn, stain_type)
        if type(dst_dir) is str:
            dst_dir = Path(dst_dir)
        self._dst_dir = dst_dir

    @property
    def cell_mask(self, ):
        return (self._dst_dir / self._naming.cell_mask).exists()

    @property
    def cell_correct_mask(self, ):
        return (self._dst_dir / self._naming.cell_correct_mask).exists()

    @property
    def tissue_mask(self, ):
        return (self._dst_dir / self._naming.tissue_mask).exists()

    @property
    def transformed_image(self, ):
        return (self._dst_dir / self._naming.transformed_image).exists()

    @property
    def registration_image(self, ):
        return (self._dst_dir / self._naming.registration_image).exists()

    @property
    def transformed_template(self, ):
        return (self._dst_dir / self._naming.transformed_template).exists()

    @property
    def stitched_template(self, ):
        return (self._dst_dir / self._naming.stitched_template).exists()


class PipelineRunState(object):
    """ implement naming management of CellBin output files and key string within the output files """

    def __init__(self, chip_no: str, dst_dir: Union[str, Path]):
        self._naming = DumpPipelineFileNaming(chip_no)
        if type(dst_dir) is str: dst_dir = Path(dst_dir)
        self._dst_dir = dst_dir

    @property
    def image_qc(self, ):  # image record file 
        return (self._dst_dir / self._naming.ipr).exists()

    @property
    def scheduler(self, ):  # image pyramid files
        flag = (self._dst_dir / self._naming.ipr).exists() and (self._dst_dir / self._naming.rpi).exists()
        return flag

    @property
    def metrics(self, ):  # statistical indicator file
        flag = self.scheduler and (self._dst_dir / self._naming.metrics).exists()
        return flag

    @property
    def report(self, ):  # report file
        flag = self.metrics and (self._dst_dir / self._naming.report).exists()
        return flag


