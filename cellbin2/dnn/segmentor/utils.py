from cellbin2.utils.common import TechType


ONNX_EXT = '.onnx'
SUPPORTED_MODELS = [
    f'cellseg_bcdu_SHDI_221008_tf',
    f'cellseg_bcdu_H_240823_tf',
    f'cellseg_unet_RNA_20230606',
    f'cellseg_bcdu_D_250225_tf_R',
    f'cellseg_bcdu_H_250303_tf_R',
]
SUPPORTED_STAIN_TYPE_BY_MODEL = {
    SUPPORTED_MODELS[0]: [TechType.ssDNA, TechType.HE, TechType.DAPI],
    SUPPORTED_MODELS[1]: [TechType.HE],
    SUPPORTED_MODELS[2]: [TechType.Transcriptomics],
    SUPPORTED_MODELS[3]: [TechType.DAPI],
    SUPPORTED_MODELS[4]: [TechType.HE]
}
weight_name_ext = '_weights_path'
TechToWeightName = {}
for key, value in SUPPORTED_STAIN_TYPE_BY_MODEL.items():
    for i in value:
        # if i in TechToWeightName:
        #     print(f"{i} skip")
        TechToWeightName[i] = i.name.upper() + weight_name_ext
