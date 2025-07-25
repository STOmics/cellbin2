# from cellbin2.matrix.matrix import cMatrix
# from pathlib import Path
# from cellbin2.contrib.param import ChipFeature
# from cellbin2.utils.common import TechType
#
#
# def matrix_feature(matrix_path: str, ref: list, tech_type: TechType) -> ChipFeature:
#     """
#     :param matrix_path:
#     :param tech_type:
#     :param ref: [[240, 300, 330, 390, 390, 330, 300, 240, 420],
#                             [240, 300, 330, 390, 390, 330, 300, 240, 420]]
#     :return:
#     """
#
#     cm = cMatrix()
#     cm.read(file_path=Path(matrix_path))
#     cm.detect_feature(ref=ref)
#     chipf = ChipFeature()
#     chipf.set_chip_box(cm.chip_box)
#     chipf.set_template(cm.template)
#     chipf.tech_type = tech_type
#     chipf.set_mat(cm.heatmap)
#
#     return chipf
#
