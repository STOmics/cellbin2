import os
import yaml
from cellbin2 import contrib
from cellbin2.contrib.template.inferencev1.inference_v1 import TemplateReferenceV1Param
from cellbin2.contrib.template.inferencev2.inference_v2 import TemplateReferenceV2Param
from cellbin2.contrib.template.inferencev1.track_line import TrackLinesParam
from cellbin2.contrib.cell_segmentor import CellSegParam
from cellbin2.contrib.tissue_segmentor import TissueSegParam
from pydantic import BaseModel, Field
from cellbin2.utils.ipr import sPlaceHolder



class DefaultIMage(BaseModel):
    clarity: bool = Field(False, description="YES default-Does the command - line input image perform the clarity identification operation?")


class Config:
    def __init__(self, config_file: str, weights_root: str = None):
        with open(config_file, 'rb') as fd:
            dct = yaml.load(fd, Loader=yaml.FullLoader)
            for module, module_v in dct.items():
                for k, v in module_v.items():
                    if 'weights_path' in k and weights_root:
                        if v is None:
                            wp = sPlaceHolder
                        else:
                            wp = os.path.join(weights_root, v)
                        dct[module][k] = wp
            self.param = dct

    @property
    def clarity(self, ):
        return contrib.ClarityParam(**self.param['clarity'])

    @property
    def track_points(self, ):
        return contrib.TrackPointsParam(**self.param['trackPoints'])

    @property
    def default_image(self, ):
        return DefaultIMage(**self.param['defaultImage'])

    @property
    def calibration(self, ):
        return contrib.CalibrationParam(**self.param['calibration'])

    @property
    def template_ref_v1(self, ):
        return TemplateReferenceV1Param(**self.param['templateReferenceV1'])

    @property
    def template_ref_v2(self, ):
        return TemplateReferenceV2Param(**self.param['templateReferenceV2'])

    @property
    def track_lines(self, ):
        return TrackLinesParam(**self.param['trackLines'])

    @property
    def chip_detector(self, ):
        return contrib.ChipParam(**self.param['chipDetector'])

    @property
    def cell_segmentation(self, ):
        return CellSegParam(**self.param['cellSegmentation'])

    @property
    def tissue_segmentation(self, ):
        return TissueSegParam(**self.param['tissueSegmentation'])

    @property
    def registration(self, ):
        from cellbin2.modules.extract.register import RegistrationParam
        return RegistrationParam(**self.param['Registration'])

    @property
    def cell_correct(self, ):
        return contrib.CellCorrect(**self.param['cellCorrect'])

    @property
    def genetic_standards(self, ):
        from cellbin2.matrix.matrix import GeneticStandards
        return GeneticStandards(**self.param['geneticStandards'])


def main():
    curr_path = os.path.dirname(os.path.realpath(__file__))
    config = Config(os.path.join(curr_path, r'../config/cellbin.yaml'))
    # cp = config.get_clarity()
    # print(cp.cluster_height_thr)
    # print(cp.ssDNA_weights_path)
    print()


if __name__ == '__main__':
    main()
