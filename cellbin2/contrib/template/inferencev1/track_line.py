import os
import cv2


from cellbin2.contrib.template.inferencev1.line_detector import TrackLineDetector
from cellbin2.image.transform import ImageTransform
from cellbin2.contrib.template.inferencev1.template_match import TemplateMatcher
from cellbin2.image import CBImage, cbimread
from cellbin2.image.augmentation import f_rgb2gray, dapi_enhance, f_gray2bgr
from pydantic import BaseModel, Field


class TrackLinesParam(BaseModel):
    score_thr: int = Field(0, description="得分阈值")
    topk: int = Field(10, description="topK")
    channel: int = Field(-1, description="topK")


def pts_on_img(img, pts, radius=1, color=(0, 0, 255), thickness=1):
    for pt in pts:
        pos = (int(pt[0]), int(pt[1]))
        cv2.circle(img, pos, radius, color, thickness)
    return img


def line_debug(arr, lines):
    h, w = arr.shape[:2]
    for line_ in lines:
        p0, p1 = line_.two_points((h, w))
        cv2.line(arr, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), (255, 0, 0), 1)
    return arr


class TrackLineQC(object):
    def __init__(self, magnification, scale_range, channel=-1, chip_template=None):
        """
        This class is used to do track line detection and template matching by using traditional algo.
        This algorithm will not work the angle of image is more than 8 degree.

        Args:
            magnification (): 10x or 20x
            scale_range (): scale search range
            channel ():
                the channel you choose to do line detect if input is multichannel
                default is -1 -> use opencv convert rgb to gray

            chip_template ():
                SS2: [
                [240, 300, 330, 390, 390, 330, 300, 240, 420],
                [240, 300, 330, 390, 390, 330, 300, 240, 420]
                ]
                default is SS2

        Result:
            self.line_result: track line detection result by traditional method
                - {'row_col': [track_lines, (image_height, image_width)]}
                - no dets: {}

            self.matching_result:
                - {'row_col': [cross_pts, x_scale, y_scale, rotation}
                - no matches: {}

        """
        self.detect_channel = channel
        if chip_template is None:
            chip_template = [
                [240, 300, 330, 390, 390, 330, 300, 240, 420],
                [240, 300, 330, 390, 390, 330, 300, 240, 420]
            ]
        self.chip_template = chip_template

        self.line_detector = TrackLineDetector()
        self.line_matcher = TemplateMatcher(magnification=magnification, scale_range=scale_range)

        self.line_result = dict()
        self.matching_result = dict()
        self.score = 0
        self.pre_func = None
        self.debug = None
        self.save_dir = None
        self.line_dir = None
        self.match_dir = None

    def set_preprocess_func(self, f):
        self.pre_func = f

    def set_debug_mode(self, d, save_dir):
        self.debug = d
        self.save_dir = save_dir
        self.line_dir = os.path.join(self.save_dir, 'line_result')
        self.match_dir = os.path.join(self.save_dir, "match_result")
        os.makedirs(self.line_dir, exist_ok=True)
        os.makedirs(self.match_dir, exist_ok=True)

    def line_detect(self, img_dict: dict, buffer=None):
        """

        Args:
            img_dict (dict): {'row_col': [img_path, angle]}, angle can be None if unknown

        """

        self.img_dict = img_dict
        self.line_result = dict()
        if isinstance(buffer, str):
            self.buffer = cbimread(buffer)

        for key, item in img_dict.items():
            img_path, angle = item
            # img_obj = CBImage()
            # read_code = img_obj.read(img_path, buffer)
            img_obj = self.buffer.crop_image(img_path)
            # if read_code != 0:
            #     continue
            if img_obj.ndim == 3:
                if self.detect_channel != -1:
                    img_obj.get_channel(0)
                else:
                    if self.pre_func is None:
                        img_obj.read(f_rgb2gray(img_obj.image.copy()))
                    else:
                        img_obj.read(f_rgb2gray(self.pre_func(img_obj)))

            ori_w, ori_h = img_obj.width, img_obj.height
            if angle is not None:
                ima_trans = ImageTransform()
                ima_trans.set_image(img_obj.image)
                new_arr = ima_trans.rot_and_crop(angle)

                new_img_obj = CBImage()
                new_img_obj.read(new_arr)
                new_w, new_h = new_img_obj.width, new_img_obj.height
                h_lines, v_lines = self.line_detector.generate(new_img_obj.image)
                track_lines = h_lines + v_lines
                for line in track_lines:
                    line.line_rotate(angle=-angle, ori_w=ori_w, ori_h=ori_h, new_w=new_w, new_h=new_h)
            else:
                h_lines, v_lines = self.line_detector.generate(img_obj.image)
                track_lines = h_lines + v_lines
            if self.debug:
                if self.pre_func is not None:
                    vis_img = f_gray2bgr(img_obj.image)
                else:
                    vis_img = dapi_enhance(img_obj)
                debug_result = line_debug(vis_img, track_lines)
                cv2.imwrite(os.path.join(self.line_dir, key + '.tif'), debug_result)
                print(f"{key}->{len(track_lines)}")
            if len(track_lines) != 0:
                self.line_result[key] = [track_lines, (ori_h, ori_w)]

    def track_match(self, ):
        """
        This func is used to match the track line results

        Returns:
            self.matching_result (dict): contain result for each fov. Including cross pts, x_scale, y_scale, rotation

        """
        self.matching_result = dict()
        # result = dict()
        if len(self.line_result) == 0:
            return
        for key, val in self.line_result.items():
            lines, shape = val
            ret_code = self.line_matcher.match(
                shape=shape,
                track_lines=lines,
                chip_template=self.chip_template
            )
            if ret_code == 0:
                self.matching_result[key] = [
                    self.line_matcher.cross_pts,
                    self.line_matcher.x_scale,
                    self.line_matcher.y_scale,
                    self.line_matcher.rotation,
                    self.line_matcher.x_cnt,
                    self.line_matcher.y_cnt,
                    self.line_matcher.x_err_min,
                    self.line_matcher.y_err_min,
                ]
                if self.debug:
                    img_path = os.path.join(self.img_dict[key][0])
                    img_obj = CBImage()
                    img_obj.read(img_path)
                    if self.pre_func is not None:
                        img_obj.read(self.pre_func(img_obj))
                        vis_img = img_obj.image
                    else:
                        vis_img = dapi_enhance(img_obj)
                    debug_result = pts_on_img(vis_img, self.line_matcher.cross_pts, radius=5, thickness=5)
                    cv2.imwrite(os.path.join(self.match_dir, key + "_template" + '.tif'), debug_result)

        self.score = len(self.matching_result) / len(self.img_dict)
        self.match_eval()

    def get_best_match(self, ):
        """
        This func is used to get the best template fov from all matching result

        Returns:
            pt (list): row_col, cross pts, x_scale, y_scale, rotation

        """
        if len(self.matching_result) == 0:
            return []
        k, v = self.matching_result[0]
        cross_pts, x_scale, y_scale, rotation, x_cnt, y_cnt, x_err_min, y_err_min = v
        pt = [k, cross_pts, x_scale, y_scale, rotation]
        return pt

    def match_eval(self, ):
        """
        This func is used to rank all matching result

        Returns:
            self.matching_result: a ranked matching result

        """
        if len(self.matching_result) == 0:
            return
        self.matching_result = sorted(
            self.matching_result.items(),
            key=lambda k: (k[-1][-4] + k[-1][-3], -k[-1][-2] + -k[-1][-1]),
            reverse=True
        )


if __name__ == '__main__':
    img_dir = r"D:\Data\qc\new_qc_test_data\qc_error\SS200000872BR_B6\SS200000872BR_B6_DAPI"
    save_dir = r"D:\Data\qc\new_qc_test_data\qc_error\SS200000872BR_B6\line_out"
    os.makedirs(save_dir, exist_ok=True)
    # angle = -1.1444091796875e-05
    angle = None
    img_dict = {name: [os.path.join(img_dir, name), angle] for name in os.listdir(img_dir)}
    track_line_qc = TrackLineQC(magnification=10, scale_range=0.8)
    track_line_qc.line_detect(img_dict)
    track_line_qc.track_match()
    print("asd")
