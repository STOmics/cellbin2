from pydantic import BaseModel, Field
from typing import List, Any, Tuple, Union
import numpy as np

from cellbin2.utils.common import TechType
from cellbin2.image import CBImage, cbimread
from cellbin2.utils.common import bPlaceHolder, fPlaceHolder, iPlaceHolder, sPlaceHolder





class TrackPointsInfo(BaseModel):
    track_points: dict = Field(dict(), description="detected track point coordinates")
    good_fov_count: int = Field(iPlaceHolder, description="number of FOV with track points above threshold ")
    score: float = Field(fPlaceHolder, description="score")
    fov_location: Any = Field(description='locations for all FOV')





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
    tissue_mask: Any = Field(None, description='outputted cell segmentation mask')
    threshold_list: Any = Field(None, description='returned threshold list')
