import cv2
import numpy as np

from typing import List, Tuple, Dict
from cellbin2.utils import clog
from pydantic import BaseModel, Field
from scipy.spatial.distance import cdist

import cellbin2.image as cbi

from cellbin2.image.augmentation import f_ij_16_to_8
from cellbin2.dnn.detector import OBB8Detector, Yolo8Detector
from cellbin2.utils.common import TechType
from typing import Union
from cellbin2.image import CBImage
from cellbin2.contrib.alignment.basic import transform_points, ChipBoxInfo
from cellbin2.contrib.base_module import BaseModule
from cellbin2.utils.plot_funcs import get_view_image

SUPPORTED_STAIN_TYPE = (TechType.ssDNA, TechType.DAPI, TechType.HE)
weight_name_ext = '_weights_path'
TechToWeightName = {i.value: i.name.lower() + weight_name_ext for i in SUPPORTED_STAIN_TYPE}


class ChipParam(BaseModel, BaseModule):
    detect_channel: int = Field(
        -1, description="If the input image is 3-channel, indicate the detection channel. "
                        "Otherwise, the program will automatically switch to a single channel diagram")
    stage1_weights_path: str = Field(
        "chip_detect_11obbn_640_stage1_20250402_pytorch.onnx",
        description="The weight file name corresponding to the first stage")
    stage2_weights_path: str = Field(
        "chip_detect_yolo11x_1024_stage2_20250411_2e3_equ_pytorch.onnx",
        description="The weight file name corresponding to the second stage")
    GPU: int = Field(0, description="The weight file name corresponding to the second stage")
    num_threads: int = Field(0, description="The number of threads used for inference")

    def get_stage1_weights_path(self, ):
        return self.stage1_weights_path

    def get_stage2_weights_path(self, ):
        return self.stage2_weights_path


class ChipDetector(object):
    """ Image data: chip area detector """

    PADDING_SIZE = 1000

    def __init__(self,
                 cfg: ChipParam,
                 stain_type: TechType):
        """
        Initialize the ChipDetector object.

        Args:
            cfg (ChipParam): Configuration parameters for the chip detector.
            stain_type (TechType): The type of stain used, which must be one of the supported types.

        Raises:
            ValueError: If the provided stain_type is not supported.

        Attributes initialized:
            cfg (ChipParam): Configuration for the chip detector.
            stain_type (TechType): The type of stain.
            chip_actual_size (Tuple[int, int]): Actual size of the chip.
            left_top, right_top, left_bottom, right_bottom (List[float]): Coordinates for the corners of the chip.
            scale_x, scale_y (float): Scaling factors for the x and y axes.
            rotation (float): Rotation angle of the chip.
            is_available (bool): Flag indicating if the chip detector is available.
            chip_size (Tuple[float, float]): Size of the chip.
            source_image: The source image for detection.
            onnx_model_global, onnx_model_local: Paths to the ONNX models for global and local detection stages.
            rough_corner_points, finetune_corner_points: Coordinates for rough and fine-tuned corner points.
            set_points_flag (bool): Flag indicating if the corner points have been set.

        Examples:
            detector = ChipDetector(cfg=my_config, stain_type=TechType.DAPI)
        """
        if stain_type not in SUPPORTED_STAIN_TYPE:
            clog.info(f"Track detect only support {[i.name for i in SUPPORTED_STAIN_TYPE]}, fail to initialize")
            return
        # Initialize configuration
        if cfg is not None:
            self.cfg: ChipParam = cfg
        else:
            self.cfg = ChipParam()

        self.stain_type = stain_type
        self.chip_actual_size = (None, None)

        # Initialize output attributes
        self.left_top: List[float] = [0.0, 0.0]
        self.right_top: List[float] = [0.0, 0.0]
        self.left_bottom: List[float] = [0.0, 0.0]
        self.right_bottom: List[float] = [0.0, 0.0]
        self.scale_x: float = 1.0
        self.scale_y: float = 1.0
        self.rotation: float = 0.0
        self.is_available: bool = True
        self.chip_size: Tuple[float, float] = (0.0, 0.0)

        self.source_image = None

        # Initialize model paths
        self.onnx_model_global = self.cfg.get_stage1_weights_path()
        self.onnx_model_local = self.cfg.get_stage2_weights_path()

        self.rough_corner_points = None
        self.finetune_corner_points = None

        self.set_points_flag = False

    def set_corner_points(self, points: np.ndarray):
        """
        Set the corner points of the chip.

        Args:
            points (np.ndarray): A numpy array of shape (4, 2) representing the corner points of the chip.

        Returns:
            None
        """
        if isinstance(points, np.ndarray):
            if points.shape == (4, 2):
                self.finetune_corner_points = points
                self.set_points_flag = True
                clog.info(f"Set corner points done.")

    def control(
            self,
            threshold_length_rate: float = 0.05,
            threshold_rotate: float = 2,
    ):
        """
        This method controls the validity of the chip detector by checking the length-to-width ratio and the included angles between consecutive corner points.

        Args:
            threshold_length_rate (float): The acceptable error rate for the length-to-width ratio.
            threshold_rotate (float): The maximum acceptable error for the included angles.

        Returns:
            None: This method updates the `is_available` attribute of the class.
        """
        self.is_available = True

        # Calculate the distance between consecutive corner points
        dist = cdist(self.finetune_corner_points, self.finetune_corner_points)
        dist_list = [dist[i, (i + 1) % dist.shape[0]] for i in range(dist.shape[0])]
        dist_rate = np.matrix([dist_list]) / np.matrix([dist_list]).T
        clog.info(f"Chip detector -> length-to-width ratio == "
                  f"max: {np.round(np.max(dist_rate), 5)}  min: {np.round(np.min(dist_rate), 5)}")

        # Expected length-to-width ratios based on the actual chip size
        _dr = (self.chip_actual_size[0] / self.chip_actual_size[1],
               self.chip_actual_size[1] / self.chip_actual_size[0])

        # Check if the actual length-to-width ratios exceed the acceptable error rate
        if np.abs(np.max(dist_rate) - np.max(_dr)) > threshold_length_rate or \
                np.abs(np.min(dist_rate) - np.min(_dr)) > threshold_length_rate:
            self.is_available = False

        # Calculate the slopes and angles between consecutive corner points
        fcp = self.finetune_corner_points
        k_list = [(fcp[i, 1] - fcp[(i + 1) % fcp.shape[0], 1]) /
                  (fcp[i, 0] - fcp[(i + 1) % fcp.shape[0], 0])
                  for i in range(fcp.shape[0])]
        r_list = list(map(lambda x: np.degrees(np.arctan(x)), k_list))

        # Calculate the included angles and check if they exceed the acceptable error
        included_angle = list()
        for i in range(len(r_list)):
            _r = r_list[(i + 1) % len(r_list)] - r_list[i]
            if _r < 0: _r = 180 + _r
            included_angle.append(_r)
        clog.info(f"Chip detector -> included angle == {list(map(lambda x: np.round(x, 5), included_angle))}")

        included_angle = np.abs(np.array(list(map(lambda x: 90 - x, included_angle))))
        if np.any(included_angle > threshold_rotate):
            self.is_available = False
        clog.info(f"Chip detector -> is available == {self.is_available}")

    def detect(self, file_path: str, actual_size: Tuple[int, int]):
        """Entry function for the detection process.

        This method is the entry point for the detection process. It initializes the detection by parsing the image,
        setting up control points, and then running the rough and fine-tune stages of detection if the points are not set.

        Args:
            file_path (str): The path to the image file to be processed.
            actual_size (Tuple[int, int]): The actual size of the chip, represented as a tuple of two integers (width, height).

        Returns:
            None: This method does not return any value.
        """
        self.parse_image(file_path, actual_size)

        if not self.set_points_flag:
            self.stage_rough()
            self.stage_finetune()

        self.parse_info()
        self.control()

    def parse_image(self, img, actual_size):
        """
        Parse the input image and adjust its size.

        Args:
            img (str or np.ndarray): The input image file path or image array.
            actual_size (Tuple[int, int]): The actual size of the chip.

        Returns:
            None
        """
        self.chip_actual_size = actual_size

        # Read the input image using CBI
        self.source_image = cbi.cbimread(img, only_np=True)
        # Convert the image from 16-bit to 8-bit
        self.source_image = f_ij_16_to_8(self.source_image)

    def parse_info(self):
        """
        This method calculates the rotation angle of the chip, its size, scale factors for the x and y axes,
        and assigns the fine-tuned corner points to the corresponding attributes.
        """

        # Calculate the rotation angle of the chip using the fine-tuned corner points
        self.rotation = self.calculate_rotation_angle(self.finetune_corner_points)

        # Calculate the chip size by finding the distances between the first and second,
        # and first and fourth fine-tuned corner points
        self.chip_size = (cdist([self.finetune_corner_points[0]], [self.finetune_corner_points[1]])[0][0],
                          cdist([self.finetune_corner_points[0]], [self.finetune_corner_points[3]])[0][0])

        # Log the calculated chip size
        clog.info('On image, chip size == {}'.format(self.chip_size))

        # Calculate the scale factors for the x and y axes based on the chip size and actual size
        _sx = np.max(self.chip_size) / np.max(self.chip_actual_size)
        _sy = np.min(self.chip_size) / np.min(self.chip_actual_size)

        # Assign the scale factors based on the comparison of the actual chip size dimensions
        if self.chip_actual_size[1] > self.chip_actual_size[0]:
            self.scale_x, self.scale_y = _sx, _sy
        else:
            self.scale_x, self.scale_y = _sy, _sx

        # Log the calculated scale factors
        clog.info('Calculate scale(XY) == ({}, {})'.format(self.scale_x, self.scale_y))

        # Assign the fine-tuned corner points to the corresponding attributes
        self.left_top, self.left_bottom, self.right_bottom, self.right_top = \
            [list(i) for i in self.finetune_corner_points]

    def stage_rough(self):
        """Perform rough detection of the chip's corner points.

        If the original image is a long rectangle (width/length < 0.9),
        it should be transformed into a square by non-proportional scaling first,

        This method initializes an OBB8Detector with the provided ONNX model and source image,
        then runs the detector to obtain the rough corner points.

        Returns:
            None. The rough corner points are stored in the `rough_corner_points` attribute.
        """

        if len(self.source_image.shape) == 3:
            h, w, _ = self.source_image.shape
        else:
            h, w = self.source_image.shape

        max_l = max(h, w)
        min_l = min(h, w)
        if min_l / max_l < 0.9:
            new_size = max_l
            square_image = cv2.resize(self.source_image, (new_size, new_size), interpolation=cv2.INTER_LINEAR)
            obb8_detector = OBB8Detector(self.onnx_model_global, square_image, gpu=self.cfg.GPU, num_threads=self.cfg.num_threads)
            square_corner_points = obb8_detector.run()
            scale_w = w / new_size
            scale_h = h / new_size
            square_corner_points[:, 0] *= scale_w  # adjust x-coordinates
            square_corner_points[:, 1] *= scale_h  # adjust y-coordinates
            self.rough_corner_points = square_corner_points
        else:
            obb8_detector = OBB8Detector(self.onnx_model_global, self.source_image, gpu=self.cfg.GPU, num_threads=self.cfg.num_threads)
            self.rough_corner_points = obb8_detector.run()

    def stage_finetune(self):
        """
        Fine-tune the corner points of the detected object.

        This method adjusts the rough corner points detected in the previous stage by:
        1. Checking the border to ensure points are within image boundaries.
        2. Calculating the rotation angle of the object.
        3. Rotating the image and transforming the corner points.
        4. Padding the image borders and adjusting the corner points accordingly.
        5. Using a YOLO detector to further refine the corner points.

        Attributes:
            rough_corner_points: The initial rough corner points detected.
            source_image: The source image being analyzed.
            PADDING_SIZE: The size of the padding to be added to the image borders.
            onnx_model_local: The ONNX model used for detection.

        Returns:
            None. The method updates the `finetune_corner_points` attribute with the refined points.
        """

        # Check the border to ensure rough corner points are within image boundaries
        self.rough_corner_points = self.check_border(self.rough_corner_points)

        # Calculate the rotation angle of the object based on rough corner points
        rotate = self.calculate_rotation_angle(self.rough_corner_points)

        # Read the source image and rotate it based on the calculated angle
        rotated_image = cbi.cbimread(self.source_image)
        rotated_image = rotated_image.trans_image(rotate=rotate)

        # Transform the corner points based on the rotation
        new_corner_points, M = transform_points(
            points=self.rough_corner_points,
            rotation=-rotate,
            src_shape=self.source_image.shape
        )
        rotated_image = rotated_image.image

        # Pad the borders of the rotated image and adjust the corner points
        rotated_image = self.padding_border(rotated_image, self.PADDING_SIZE)
        new_corner_points += self.PADDING_SIZE

        # Initialize a list to store the refined points
        new_points = list()

        # Loop through each corner point to refine it using YOLO detection
        for i, _p in enumerate(new_corner_points):
            x, y = map(lambda k: self.PADDING_SIZE if k < self.PADDING_SIZE else int(k), _p)

            # Ensure the points do not go out of the padded image boundaries
            if x > rotated_image.shape[1] - self.PADDING_SIZE: x = rotated_image.shape[1] - self.PADDING_SIZE
            if y > rotated_image.shape[0] - self.PADDING_SIZE: y = rotated_image.shape[0] - self.PADDING_SIZE

            # Extract a patch around the current point from the rotated image
            _img = rotated_image[y - self.PADDING_SIZE: y + self.PADDING_SIZE,
                   x - self.PADDING_SIZE: x + self.PADDING_SIZE]

            # Initialize and run the YOLO detector on the image patch
            yolo8_detector = Yolo8Detector(self.onnx_model_local, _img, gpu=self.cfg.GPU, num_threads=self.cfg.num_threads)
            yolo8_detector.set_preprocess_func(self._finetune_preprocess)

            points = yolo8_detector.run()
            points = self.check_border(points)
            new_points.append(points[i] - self.PADDING_SIZE)
            new_corner_points[i, :] = [x, y]

        # Calculate the final refined corner points and update the attribute
        finetune_points = np.array(new_points) + new_corner_points
        finetune_points -= self.PADDING_SIZE
        self.finetune_corner_points = self._inv_points(M, finetune_points)

        # TODO: Transform the four corner points into a rectangle in the future

    @staticmethod
    def _finetune_preprocess(img):
        """
        Preprocess the image for fine-tuning by equalizing the histogram and converting it to RGB format.

        Parameters:
        img (numpy.ndarray): The input image to preprocess.

        Returns:
        numpy.ndarray: The preprocessed image in RGB format.
        """
        if img.ndim == 3:
            ei = cv2.equalizeHist(img[:, :, 0])
        else:
            ei = cv2.equalizeHist(img)

        ei = cv2.cvtColor(ei, cv2.COLOR_GRAY2RGB)

        return ei

    @staticmethod
    def _inv_points(mat, points):
        """
        Inverts the given transformation matrix and applies it to the provided points.

        Args:
            mat (np.ndarray): The transformation matrix to be inverted.
            points (np.ndarray): The points to be transformed.

        Returns:
            np.ndarray: The transformed points after applying the inverted matrix.
        """
        if mat.shape == (2, 3):
            _mat = np.eye(3)
            _mat[:2, :] = mat
        else:
            _mat = mat

        new_points = np.matrix(_mat).I @ np.concatenate(
            [points, np.ones((points.shape[0], 1))], axis=1
        ).transpose(1, 0)

        return np.array(new_points)[:2, :].transpose()

    @staticmethod
    def calculate_rotation_angle(points):
        """
        Calculate the angle between the line segment joining the two points with the smallest y-coordinates
        and the horizontal axis.

        Parameters:
        points (list of tuples): List of points in the format (x, y).

        Returns:
        float: The angle in degrees between the line segment and the horizontal axis.
        """

        # Sort points based on y-coordinate
        sorted_points = sorted(points, key=lambda p: p[1])

        # Select the two points with the smallest y-coordinates
        y_min_points = sorted_points[:2]

        # From the two points with the smallest y-coordinates, select the one with the minimum x-coordinate
        # and the one with the maximum x-coordinate
        p1 = min(y_min_points, key=lambda p: p[0])  # Point with the minimum x-coordinate
        p2 = max(y_min_points, key=lambda p: p[0])  # Point with the maximum x-coordinate

        # Calculate the angle of the line segment p1p2
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        # Calculate the angle using arctan2 which handles the quadrant of the point correctly
        angle = np.arctan2(dy, dx)

        # Convert the angle from radians to degrees
        angle_degrees = np.degrees(angle)

        return angle_degrees

    @staticmethod
    def padding_border(img, size):
        """
        Pads with the pixel mean values of the first three columns and the bottom four rows.

        Parameters:
        img (numpy.ndarray): The input image to be padded.
        size (int): The size of the border to be added to each side of the image.

        Returns:
        numpy.ndarray: The padded image.
        """

        if len(img.shape) == 3:
            top_pixels = img[:3, :, :]
            left_pixels = img[:, :3, :]
            bottom_pixels = img[-3:, :, :]
            right_pixels = img[:, -3:, :]
            combined_pixels = np.vstack([
                top_pixels.reshape(-1, 3),
                left_pixels.reshape(-1, 3),
                bottom_pixels.reshape(-1, 3),
                right_pixels.reshape(-1, 3)
            ])
            mean_value = tuple(int(x) for x in np.mean(combined_pixels, axis=0))
        elif len(img.shape) == 2:
            top_pixels = img[:3, :].flatten()
            left_pixels = img[:, :3].flatten()
            bottom_pixels = img[-3:, :].flatten()
            right_pixels = img[:, -3:].flatten()
            mean_value = int(np.mean(np.concatenate([top_pixels, left_pixels, bottom_pixels, right_pixels])))

        return cv2.copyMakeBorder(img, size, size, size, size, cv2.BORDER_CONSTANT, value=mean_value)

    @staticmethod
    def check_border(file: np.ndarray):
        """
        Validate and reorder an array of corner points.

        The function checks if the input is a NumPy array with the shape (4, 2) and reorders the points to ensure
        they follow the sequence: left-up, left-down, right-down, right-up. If the points are out of order, they are
        reordered accordingly.

        Args:
            file (np.ndarray): An array of corner points with shape (4, 2).

        Returns:
            np.ndarray: The validated and reordered array of corner points, or None if the input is invalid.
        """
        if not isinstance(file, np.ndarray): return None
        assert file.shape == (4, 2), "Array shape error."

        file = file[np.argsort(np.mean(file, axis=1)), :]
        if file[1, 0] > file[2, 0]:
            file = file[(0, 2, 1, 3), :]

        file = file[(0, 1, 3, 2), :]

        return file


def detect_chip(file_path: Union[str, np.ndarray],
                cfg: ChipParam,
                stain_type: TechType,
                actual_size: Tuple[int, int],
                is_debug: bool) -> Tuple[ChipBoxInfo, Dict[str, np.ndarray]]:
    """
    Detects a chip in the given image file and calculates its parameters.

    Args:
        file_path: The path to the image file or a numpy array containing the image.
        cfg: Configuration parameters for the chip detection.
        stain_type: The type of staining used (ssDNA, DAPI, HE, IF).
        actual_size: The actual size of the chip in pixels at 500nm/pixel resolution (width, height).
        is_debug: Flag to enable or disable debug mode.

    Returns:
        A tuple containing ChipBoxInfo with detection results and a dictionary of debug images if is_debug is True.
    """
    debug_image_dic = {}
    cd = ChipDetector(cfg=cfg, stain_type=stain_type)
    cd.detect(file_path, actual_size=actual_size)
    info = {
        'LeftTop': cd.left_top, 'LeftBottom': cd.left_bottom,
        'RightTop': cd.right_top, 'RightBottom': cd.right_bottom,
        'ScaleX': cd.scale_x, 'ScaleY': cd.scale_y, 'Rotation': cd.rotation,
        'ChipSize': cd.chip_size, 'IsAvailable': cd.is_available
    }
    # If is_debug is False, the returned dictionary debug_image_dic is empty
    if is_debug:
        points = np.array([cd.left_top, cd.left_bottom, cd.right_bottom, cd.right_top])
        debug_image_dic = get_view_image(image=file_path,
                                         points=points,
                                         is_matrix=False,
                                         radius=50)
    return ChipBoxInfo(**info), debug_image_dic


def main():
    cfg = ChipParam(
        **{"stage1_weights_path":
               r"D:\hedongdong1\Workspace\01.chip_box_detect\algorithm_develop\old_model\chip_detect_obb8n_640_SD_202409_pytorch.onnx",
           "stage2_weights_path":
               r"D:\hedongdong1\Workspace\01.chip_box_detect\algorithm_develop\old_model\chip_detect_yolo8x_1024_SDH_stage2_202410_pytorch.onnx"})

    file_path = r"D:\hedongdong1\Workspace\01.chip_box_detect\show_interface\test_data\C04144G513_ssDNA_stitch.tif"
    info, debug_dic = detect_chip(file_path, cfg=cfg, stain_type=TechType.ssDNA, actual_size=(19992, 19992),
                                  is_debug=True)
    print(info.IsAvailable)


if __name__ == '__main__':
    # points = np.loadtxt(r"D:\02.data\temp\temp_cellbin2_test\trans_data_1\B03025E4\B03025E4_DAPI_stitch.txt")
    #
    # cd = ChipDetector(cfg = None, stain_type = "DAPI")
    # cd.set_corner_points(points)
    # cd.detect(r"D:\02.data\temp\temp_cellbin2_test\trans_data_1\A00792D3\label.tif", (19992, 19992))

    main()
