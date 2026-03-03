# from .template_reference_v1 import TemplateReferenceV1
# from .template_reference_v2 import TemplateReferenceV2
from pydantic import BaseModel, Field
from .template import inference

from cellbin2.contrib.template.point_detector import TrackPointsParam
from .clarity import ClarityParam
from .chip_detector import ChipParam
from .calibration import CalibrationParam
# from cellbin2.contrib.template.inferencev2.inference_v2 import TemplateReferenceV2Param
# from cellbin2.contrib.template.inferencev1.inference_v1 import TemplateReferenceV1Param
# from cellbin2.contrib.template.inferencev1.track_line import TrackLinesParam
# from .param import WeightParam
# from .param import StitchingTemplateParam
# from .param import TrackPointsParam
# from .param import TrackLinesParam
# from .param import StitchingParam
# from .param import RegistrationParam
# from .param import CellCorrectParam
# from .param import ClarityParam


class CellCorrect(BaseModel):
    process: int = Field(5, description="number of processes")
