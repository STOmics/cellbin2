import json
import os
import re
from typing import List, Dict, Tuple, Any, Type, Union
import numpy as np
from objtyping import objtyping
import h5py

from cellbin2.utils import h52dict, obj2dict
from cellbin2.utils.common import bPlaceHolder, fPlaceHolder, iPlaceHolder, sPlaceHolder
from cellbin2.utils import HDF5
from cellbin2.contrib import param
from cellbin2.contrib import alignment
from cellbin2.contrib.alignment.basic import ChipBoxInfo
from cellbin2.contrib.template.inference import TemplateInfo
from cellbin2.contrib.alignment import RegistrationOutput
from cellbin2.contrib.clarity import ClarityOutput
from cellbin2.contrib.alignment import Registration00Output, Registration00Offset

IPR_VERSION = '0.3.0'
ALLOWED = [int, str, float, bool, list, np.int64, np.int32, np.float64, np.ndarray, np.bool_, tuple]


class BaseIpr:
    def get_attrs(self):
        keep = {}
        for k, v in self.__dict__.items():
            if isinstance(v, tuple(ALLOWED)):
                keep[k] = v
        return keep


class ImageInfo(BaseIpr):
    def __init__(self) -> None:
        self.AppFileVer: str = sPlaceHolder
        self.BackgroundBalance: bool = bPlaceHolder
        self.BitDepth: int = iPlaceHolder
        self.Brightness: int = iPlaceHolder
        self.ChannelCount: int = iPlaceHolder
        self.ColorEnhancement: bool = bPlaceHolder
        self.Contrast: bool = bPlaceHolder
        self.DeviceSN: str = sPlaceHolder  # optional
        self.DistortionCorrection: bool = bPlaceHolder
        self.ExposureTime: float = fPlaceHolder
        self.FOVHeight: int = iPlaceHolder
        self.FOVWidth: int = iPlaceHolder
        self.Gain: str = sPlaceHolder
        self.Gamma: float = fPlaceHolder
        self.GammaShift: bool = bPlaceHolder
        self.Illuminance: str = sPlaceHolder
        self.Manufacturer: str = sPlaceHolder
        self.Model: str = sPlaceHolder
        self.Overlap: float = fPlaceHolder
        # self.Pitch: int = 1
        self.PixelSizeX: float = fPlaceHolder
        self.PixelSizeY: float = fPlaceHolder
        self.QCResultFile: str = sPlaceHolder  # optional
        self.RegisterVersion: str = sPlaceHolder
        self.RGBScale: np.ndarray = np.array([iPlaceHolder, iPlaceHolder, iPlaceHolder])
        self.STOmicsChipSN: str = sPlaceHolder
        self.ScanChannel: str = sPlaceHolder
        self.ScanCols: int = iPlaceHolder
        self.ScanObjective: float = fPlaceHolder
        self.ScanRows: int = iPlaceHolder
        self.ScanTime: str = sPlaceHolder
        self.Sharpness: float = fPlaceHolder
        self.StitchedImage: bool = bPlaceHolder
        self.WhiteBalance: str = sPlaceHolder
        self.STOmicsChipFovCol: str = sPlaceHolder
        self.STOmicsChipFovRow: str = sPlaceHolder


class QCInfo(BaseIpr):
    def __init__(self):
        self.ClarityPreds: np.ndarray = np.array([])  # add at 2024/10/24, by @dzh
        self.ClarityCounts: str
        self.ClarityCutSize: List[int, int] = [iPlaceHolder, iPlaceHolder]
        self.ClarityOverlap: int = iPlaceHolder
        self.ClarityScore: int = iPlaceHolder
        self.Experimenter: str = sPlaceHolder
        self.GoodFovCount: int = iPlaceHolder
        self.ImageQCVersion: str = sPlaceHolder
        self.QCPassFlag: int = iPlaceHolder
        self.RemarkInfo: str = sPlaceHolder
        self.StainType: str = sPlaceHolder
        self.TemplateRecall: float = fPlaceHolder
        self.TemplateValidArea: float = fPlaceHolder
        self.TotalFovCount: int = iPlaceHolder
        self.TrackCrossQCPassFlag: int = iPlaceHolder
        self.ChipDetectQCPassFlag: int = iPlaceHolder
        self.TrackLineChannel: int = iPlaceHolder
        self.TrackLineScore: int = iPlaceHolder
        self.CrossPoints: CrossPoints = CrossPoints()
        self.ChipBBox: ChipBBox = ChipBBox()

    def update_clarity(self, clarity_out: ClarityOutput):
        self.ClarityCutSize = clarity_out.cut_siz
        self.ClarityOverlap = clarity_out.overlap
        self.ClarityScore = clarity_out.score
        self.ClarityPreds = clarity_out.pred


class ChipBBox(object):
    # update at 2024.10.08
    def __init__(self):
        self.LeftTop: np.ndarray = np.array([fPlaceHolder, fPlaceHolder], dtype=float)
        self.LeftBottom: np.ndarray = np.array([fPlaceHolder, fPlaceHolder], dtype=float)
        self.RightTop: np.ndarray = np.array([fPlaceHolder, fPlaceHolder], dtype=float)
        self.RightBottom: np.ndarray = np.array([fPlaceHolder, fPlaceHolder], dtype=float)
        self.ScaleX: float = fPlaceHolder
        self.ScaleY: float = fPlaceHolder
        self.ChipSize: np.ndarray = np.array([fPlaceHolder, fPlaceHolder], dtype=float)
        self.Rotation: float = fPlaceHolder
        self.IsAvailable: bool = bPlaceHolder

    def update(self, box: ChipBoxInfo):
        for k, v in box.model_dump().items():
            setattr(self, k, v)

    def get(self):
        info = ChipBoxInfo(**self.__dict__)
        return info


class CrossPoints(object):

    def __init__(self):
        pass

    def add_dataset(self, name: str, dataset: np.ndarray):
        if not hasattr(self, name):
            self.__setattr__(name, dataset)

    def add_points(self, points: Dict[str, np.ndarray]):
        for k, v in points.items():
            self.add_dataset(k, v)

    @property
    def group_points(self, ):
        dct = {}
        for k, v in vars(self).items():
            dct[k] = v
        return dct

    @property
    def stack_points(self, ):
        points = np.ones([0, 4])
        for _, v in vars(self).items():
            points = np.concatenate((points, v), axis=0)
        return points

    def clear(self, ):
        for k in vars(self).keys():
            self.__delattr__(k)


class Register00OffsetInfo:
    def __init__(self):
        self.offset: List[float] = [fPlaceHolder, fPlaceHolder]
        self.dist: float = fPlaceHolder

    def update(self, offset: Registration00Offset):
        self.offset = offset.offset
        self.dist = offset.dist


class Register00:
    def __init__(self):
        self.rot0 = Register00OffsetInfo()
        self.rot90 = Register00OffsetInfo()
        self.rot180 = Register00OffsetInfo()
        self.rot270 = Register00OffsetInfo()

    def update(self, output: Registration00Output):
        for i, v in iter(output):
            getattr(self, i).update(v)

    def get(self) -> Registration00Output:
        info = Registration00Output(**obj2dict(self))
        return info


class RegisterInfo(BaseIpr):
    def __init__(self):
        self.OffsetX: float = fPlaceHolder
        self.OffsetY: float = fPlaceHolder
        self.Flip: bool = bPlaceHolder
        self.Method: str = sPlaceHolder  # update at 2024-10-17, TemplateCentroid/Template00Pt/ChipBox
        self.CounterRot90: int = iPlaceHolder
        self.MatrixShape: List[int] = []

    def update(self, info: RegistrationOutput):
        self.OffsetX, self.OffsetY = info.offset
        self.Flip = info.flip
        self.RegisterScore = info.register_score
        self.Method = info.method.name
        self.CounterRot90 = info.counter_rot90
        self.MatrixShape = info.dst_shape


class Register(RegisterInfo):
    def __init__(self):
        super(Register, self).__init__()
        self.Rotation: float = fPlaceHolder
        self.ScaleX: float = fPlaceHolder
        self.ScaleY: float = fPlaceHolder
        self.XStart: int = iPlaceHolder
        self.YStart: int = iPlaceHolder
        self.MatrixTemplate: np.ndarray = np.array([])
        self.RegisterTemplate: np.ndarray = np.array([])
        self.RegisterTrackTemplate: np.ndarray = np.array([])
        self.GeneChipBBox = ChipBBox()
        self.Register00 = Register00()
        self.RegisterChip = RegisterInfo()


class TissueSeg(object):
    def __init__(self):
        self.TissueMask: np.ndarray = np.array([], dtype=np.int64)
        self.TissueSegScore: int = 0
        self.TissueSegShape: List[int] = []


class CellSeg(object):
    def __init__(self) -> None:
        self.CellMask: np.ndarray = np.array([], dtype=np.int64)
        self.CellSegTrace: np.ndarray = np.array([], dtype=np.int32)
        self.CellSegShape: List[int] = []


class ScopeStitch(object):
    def __init__(self):
        self.GlobalHeight: int = iPlaceHolder
        self.GlobalWidth: int = iPlaceHolder
        self.GlobalLoc: np.ndarray = np.array([], dtype=np.int64)


class StitchEval(object):
    def __init__(self):
        self.MaxDeviation: float = fPlaceHolder
        self.GlobalDeviation: np.ndarray = np.array([], dtype=float)


class Stitch(object):
    def __init__(self):
        self.TemplateSource: str = sPlaceHolder
        self.WhichStitch: str = sPlaceHolder
        self.TemplatePoint: np.ndarray = np.array([])  # stitch template
        self.TrackPoint: np.ndarray = np.array([])  # stitch track detect point
        self.TransformTemplate: np.ndarray = np.array([])  # transform template
        self.TransformTrackPoint: np.ndarray = np.array([])  # transform track detect point
        self.TransformChipBBox = ChipBBox()
        self.ScopeStitch = ScopeStitch()
        self.StitchEval = StitchEval()
        self.TransformShape = (iPlaceHolder, iPlaceHolder)


class Scope(object):
    def __init__(self):
        self.Confidence: float = fPlaceHolder
        self.OffsetX: float = fPlaceHolder
        self.OffsetY: float = fPlaceHolder


class Calibration(object):
    def __init__(self):
        self.CalibrationQCPassFlag: np.int64 = iPlaceHolder
        self.Scope = Scope()

    def update(self, r: param.CalibrationInfo):
        self.CalibrationQCPassFlag = r.pass_flag
        self.Scope.OffsetX, self.Scope.OffsetY = r.offset
        self.Scope.Confidence = r.score


class ManualState(object):
    def __init__(self):
        self.calibration: bool = False
        self.cellseg: bool = False
        self.register: bool = False
        self.stitch: bool = False
        self.tissueseg: bool = False


class StereoResepSwitch(object):
    def __init__(self):
        self.cellseg: bool = False
        self.register: bool = False
        self.stitch: bool = False
        self.tissueseg: bool = False


class ImageChannel(HDF5):
    def __init__(self):
        super(ImageChannel).__init__()
        self.CellSeg = CellSeg()
        self.ImageInfo = ImageInfo()
        self.QCInfo = QCInfo()
        self.Register = Register()
        self.Stitch = Stitch()
        self.TissueSeg = TissueSeg()

    def update_template_points(self, points_info: param.TrackPointsInfo, template_info: TemplateInfo):
        self.QCInfo.CrossPoints.add_points(points_info.track_points)
        self.QCInfo.GoodFovCount = points_info.good_fov_count
        self.QCInfo.TrackLineScore = points_info.score

        # self.QCInfo.QcPassFlag = 1
        self.QCInfo.TemplateRecall = template_info.template_recall
        self.QCInfo.TemplateValidArea = template_info.template_valid_area
        self.QCInfo.TrackCrossQCPassFlag = template_info.trackcross_qc_pass_flag
        self.QCInfo.TrackLineChannel = template_info.trackline_channel
        self.Register.Rotation = template_info.rotation
        self.Register.ScaleX = template_info.scale_x
        self.Register.ScaleY = template_info.scale_y
        self.Stitch.TemplatePoint = np.array(template_info.template_points)

    def update_registration(self, info: RegistrationOutput):
        self.Register.update(info)

    def get_registration(self, ) -> RegistrationOutput:
        r = RegistrationOutput(
            counter_rot90=self.Register.CounterRot90, flip=self.Register.Flip,
            register_score=self.Register.RegisterScore,
            offset=(self.Register.OffsetX, self.Register.OffsetY), dst_shape=self.Register.MatrixShape,
        )
        return r

    def update_basic_info(self, chip_name: str, channel: int, depth: int, height: int, width: int, stain_type: str):
        self.ImageInfo.ChannelCount = channel
        self.ImageInfo.BitDepth = depth
        self.Stitch.ScopeStitch.GlobalHeight = height
        self.Stitch.ScopeStitch.GlobalWidth = width
        self.QCInfo.StainType = stain_type
        self.ImageInfo.STOmicsChipSN = chip_name

    @property
    def box_info(self, ):
        cb = self.QCInfo.ChipBBox.get()
        # cbi = param.ChipBoxInfo(
        #     left_top=cb.LeftTop, left_bottom=cb.LeftBottom, right_top=cb.RightTop,
        #     right_bottom=cb.RightBottom, scale_x=cb.ScaleX, scale_y=cb.ScaleY,
        #     chip_size=cb.ChipSize,
        #     rotation=cb.Rotation, is_available=cb.IsAvailable)

        return cb

    @property
    def stitched_template_info(self, ):
        ti = TemplateInfo(template_recall=self.QCInfo.TemplateRecall,
                          template_valid_area=self.QCInfo.TemplateValidArea,
                          trackcross_qc_pass_flag=self.QCInfo.TrackCrossQCPassFlag,
                          trackline_channel=self.QCInfo.TrackLineChannel,
                          rotation=self.Register.Rotation,
                          scale_x=self.Register.ScaleX, scale_y=self.Register.ScaleY,
                          template_points=self.Stitch.TemplatePoint)

        return ti

    @property
    def transform_template_info(self, ):
        ti = TemplateInfo(template_recall=1,
                          template_valid_area=1,
                          trackcross_qc_pass_flag=1,
                          rotation=0,
                          scale_x=1, scale_y=1,
                          points=self.Stitch.TransformTemplate)

        return ti


class IFChannel(ImageChannel):
    def __init__(self):
        super(IFChannel, self).__init__()
        self.Calibration = Calibration()
        self.CellSeg = CellSeg()
        self.ImageInfo = ImageInfo()
        self.QCInfo = QCInfo()
        self.Register = Register()
        self.Stitch = Stitch()
        self.TissueSeg = TissueSeg()


class ImageProcessRecord(HDF5):

    def __init__(self):
        super(ImageProcessRecord).__init__()

        self.IPRVersion = IPR_VERSION
        self.ManualState = ManualState()
        self.Preview: np.ndarray = np.array([], dtype=np.uint8)
        self.StereoResepSwitch = StereoResepSwitch()


def write_channel_image(file_path, image: Union[IFChannel, ImageChannel]):
    assert file_path.name.endswith('.ipr'), '{}, expected file suffix is .ipr'.format(os.path.basename(file_path))
    image.write(file_path, extra={})
    return 0


def write(file_path, ipr: ImageProcessRecord, extra_images: dict = None):
    assert file_path.name.endswith('.ipr'), '{}, expected file suffix is .ipr'.format(os.path.basename(file_path))

    if extra_images is None:
        extra_images = dict()
    ipr.write(file_path, extra_images)
    return 0


def read(file_path: str) -> Tuple[ImageProcessRecord, Dict[str, Union[IFChannel, ImageChannel]]]:
    dct = {}
    with h5py.File(file_path, 'r') as fd:
        h52dict(fd, dct)
    ipr_dct = {}
    image_dct: Dict[str, Union[IFChannel, ImageChannel]] = {}

    for k, v in dct.items():
        if k in ['IPRVersion', 'ManualState', 'Preview', 'StereoResepSwitch']:
            ipr_dct[k] = v
        elif 'Calibration' in v.keys():
            image_dct[k] = objtyping.from_primitive(v, IFChannel)
        else:
            image_dct[k] = objtyping.from_primitive(v, ImageChannel)
    ipr = objtyping.from_primitive(ipr_dct, ImageProcessRecord)

    return ipr, image_dct


def read_key_metrics(file_path: str) -> dict:
    dct = {}
    with h5py.File(file_path, 'r') as fd:
        h52dict(fd, dct)
    image_dct = {}

    dct = {k: v for k, v in dct.items() if k not in ['IPRVersion', 'ManualState', 'Preview', 'StereoResepSwitch']}
    for channel_name, channel_data in dct.items():
        image_dct[channel_name] = {k: v for k, v in channel_data.items() if k in ['ImageInfo', 'QCInfo', 'Register']}

    return image_dct


def main():
    # ipr = ImageProcessRecord()
    # images = {'IF1': IFChannel(), 'DAPI': ImageChannel()}
    # images['DAPI'].QCInfo.StainType = TechType.DAPI.name
    #
    # points = {'0_0': np.array([[0, 0, 0, 0],
    #                      [1, 1, 1, 1],
    #                      [2, 2, 2, 2]], dtype=float),
    #           '2_0': np.array([[0, 0, 8, 0],
    #                          [1, 1, 1, 1],
    #                          [2, 2, 5, 2]], dtype=float)
    # }
    # images['DAPI'].QCInfo.CrossPoints.add_points(points)
    # images['DAPI'].QCInfo.CrossPoints.add_dataset(
    #     '1_0', np.array([[0, 0, 0, 0],
    #                      [1, 1, 1, 1],
    #                      [2, 2, 2, 2]], dtype=float))
    # images['DAPI'].Stitch.TemplatePoint = np.array([[0, 2, 3, 4], [7, 8, 9, 0]], dtype=int)
    # images['IF1'].QCInfo.CrossPoints.add_dataset(
    #     '2_0', np.array([[0, 0, 0, 0],
    #                      [1, 1, 1, 1],
    #                      [2, 2, 2, 2]], dtype=float))
    # file_path = "/media/Data/dzh/data/cellbin2/test/SS200000135TL_D1_demo_1/SS200000135TL_D1.ipr"
    # ipr, image_dct = read(file_path)
    #
    ifc = ImageChannel()
    ifc.Register.Register00.get()
    ifc.write("/media/Data1/user/dengzhonghan/data/cellbin2/random_test/A03599D1_11/tmp.ipr", extra={})
    # ifc.box_info()
    print()
    # dct = read_key_metrics(file_path)
    # write(file_path, ipr, images)

    # ipr, images = read(file_path)
    # print(ipr.IPRVersion)
    # for k in images.keys():
    #     print(k, type(images[k]))
    # print(images['A03599D1_DAPI'].QCInfo.CrossPoints.group_points)


if __name__ == '__main__':
    main()
