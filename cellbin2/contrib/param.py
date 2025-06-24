from pydantic import BaseModel, Field
from typing import List, Any, Tuple, Union
import numpy as np

from cellbin2.utils.common import TechType
from cellbin2.image import CBImage, cbimread
from cellbin2.utils.common import bPlaceHolder, fPlaceHolder, iPlaceHolder, sPlaceHolder





class TrackPointsInfo(BaseModel):
    track_points: dict = Field(dict(), description="识别到的track点坐标信息")
    good_fov_count: int = Field(iPlaceHolder, description="track点数目在某个数目之上的FOV数目")
    score: float = Field(fPlaceHolder, description="得分")
    fov_location: Any = Field(description='全部FOV坐标')





class CalibrationInfo(BaseModel):
    score: float = Field(fPlaceHolder, description='')
    offset: List[float] = Field([fPlaceHolder, fPlaceHolder], description='')
    scale: float = Field(fPlaceHolder, description='')
    rotate: float = Field(fPlaceHolder, description='')
    pass_flag: int = Field(iPlaceHolder, description='')





class CellSegInfo(BaseModel):
    mask: Any = Field(None, description='')
    fast_mask: Any = Field(None, description='')


class TissueSegOutputInfo(BaseModel):
    tissue_mask: Any = Field(None, description='输出的组织分割mask')
    threshold_list: Any = Field(None, description='返回的阈值列表')
