from cellbin2.utils.common import TechType
from cellbin2.utils import clog


class SupportModel(object):
    def __init__(self):

        clog.info('SupportModel init start')
        self.ONNX_EXT = '.onnx'
        self.SUPPORTED_MODELS = [
            'tissueseg_bcdu_SDI_230523_tf',   # DAPI correspond V230523 common model for ssDNA and DAPI
            'tissueseg_bcdu_S_240618_tf',
            'tissueseg_bcdu_H_20240201_tf',
            'tissueseg_bcdu_H_20241018_tf',   # newest HE R&D version V241018
            'tissueseg_bcdu_rna_220909_tf',    # rna model
            '-'
        ]
        self.SUPPORTED_STAIN_TYPE_BY_MODEL = {
            self.SUPPORTED_MODELS[0]: [TechType.ssDNA, TechType.DAPI],
            self.SUPPORTED_MODELS[1]: [TechType.ssDNA],
            self.SUPPORTED_MODELS[2]: [TechType.HE],
            self.SUPPORTED_MODELS[3]: [TechType.HE],
            self.SUPPORTED_MODELS[4]: [TechType.Transcriptomics, TechType.Protein],
            self.SUPPORTED_MODELS[5]: [TechType.IF]

        }
        self.WEIGHT_NAME_EXT = '_weights_path'
        self.TECH_WEIGHT_NAME = {}
        for key, value in self.SUPPORTED_STAIN_TYPE_BY_MODEL.items():
            for i in value:
                if i not in self.TECH_WEIGHT_NAME:
                    self.TECH_WEIGHT_NAME[i] = i.name.lower() + self.WEIGHT_NAME_EXT
                # else:
                #     clog.info(f"{i} skip")
        # for key, value in self.TECH_WEIGHT_NAME.items():
        #     clog.info(f"{key}:{value}")
        clog.info('SupportModel init complete')
