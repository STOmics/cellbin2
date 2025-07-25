from cellbin2.contrib.template.point_detector import TrackPointQC
from cellbin2.contrib.template.inferencev1.track_line import TrackLineQC
from cellbin2.utils import clog
from cellbin2.image.augmentation import dapi_enhance


class RegistQC(object):
    def __init__(
            self,
            topk=10,
    ):
        """

        Args:
            enhance_func (): enhance method used to preprocess image
            model_path (): detect weight path
            scope_type: 10x or 20x
            scale_range (): scale search range
            topk: topk track point result will be sent to line detect algo

            pt_detect_channel ():
                the channel you choose to do track detect if input is multichannel
                default is -1 -> just use the original image
            line_detect_channel ():

                the channel you choose to do line detect if input is multichannel
                default is -1 -> use opencv convert rgb to gray if multichannel

            chip_template ():
                SS2: [
                [240, 300, 330, 390, 390, 330, 300, 240, 420],
                [240, 300, 330, 390, 390, 330, 300, 240, 420]
                ]
                default is SS2

            th (): [th, th2) -> track_point_score = 1
            th2 (): [th2, inf) -> track_point_score = 2
                - use default value as an example
                - [0, 5) -> fov is scored as 0
                - [5, 20) -> fov is scored as 1
                - [20, inf) -> fov is scored as 2

        Result:
            self.pt_result: track point detection result by deep learning method
                - {'row_col': [[pt_x, pt_y, conf], angle]}
                - no dets: {}

            self.pt_score: track_pts eval score, score interval is [0, 1], higher -> better
                - int
                - Recommended score threshold for dapi is 0.4?

            self.line_result:  track line detection result by traditional method
                - {'row_col': [track_lines, (image_height, image_width)]} (10x)
                - {'image_name': [track_lines, (image_height, image_width)]} (20x)
                - no dets: {}

            self.line_score: count of matched fovs / length of given fovs
                - as long as this score is greater than 0, you will get at least one template

        """
        self._magnification = None
        self._pt_detect_channel = None
        self._line_detect_channel = None
        self._chip_template = None
        self.track_pt_qc = TrackPointQC()
        self.topk = topk

        self.pt_result = {}
        self.pt_score = 0
        self.line_result = {}
        self.line_score = 0
        self.best_match = []
        self.fovs = {}
        self.good_fov_count = 0

    def set_chip_template(self, c):
        self._chip_template = c

    def set_topk(self, k):
        self.topk = k

    def set_track_pt_thresh(self, th, th2, good_thresh):
        self.track_pt_qc.set_threshold(th, th2, good_thresh)

    def set_track_pt_process(self, p):
        self.track_pt_qc.set_multi_process(p)

    def run_pt_qc(self, fovs, enhance_func, detect_channel=-1, buffer=None):
        """
        This func is used to run track point detection on fovs

        Args:
            fovs (): 'row_col': img_path}

        Returns:
            1: fail
            0: success

        """
        self.track_pt_qc.set_detect_channel(detect_channel)
        if enhance_func is None:
            enhance_func = dapi_enhance
        self.track_pt_qc.set_enhance_func(enhance_func)
        clog.info(f"Track pt detect using enhance func {enhance_func}, using channel {detect_channel}")
        self.pt_result = {}
        self.pt_score = 0
        self.line_result = {}
        self.line_score = -1
        self.good_fov_count = 0
        self.fovs = fovs

        if len(fovs) == 0:
            return 1

        self.track_pt_qc.track_detect(fovs, buffer)
        self.track_pt_qc.track_eval()

        if self.track_pt_qc.score == 0:
            return 1

        self.pt_result = self.track_pt_qc.track_result
        self.pt_score = self.track_pt_qc.score
        self.good_fov_count = self.track_pt_qc.good_fov_count

        return 0

    def run_line_qc(self, line_fovs=None, detect_channel=-1, magnification='10x', buffer=None, enhance_func=None):
        """
        This func is used to detect track line on fovs

        Args:
            line_fovs ():
                10x -> None -> choose topk fov from track qc result
                20x -> given by stitch module, {"image_name": image (np.ndarray)}

        Returns:
            1: fail
            0: success

        """
        track_ln_pc = TrackLineQC(
            magnification=magnification,
            scale_range=0.8,
            channel=detect_channel,
            chip_template=self._chip_template,
        )

        track_ln_pc.set_preprocess_func(enhance_func)
        clog.info(f"Track line detect using enhance func {enhance_func}, using channel {detect_channel}")
        if line_fovs is None:
            line_fovs = {}
            length = len(self.track_pt_qc.fovs_order)
            if length == 0:
                return 1
            self.topk = min(self.topk, length)

            for k in range(self.topk):
                fov_name = self.track_pt_qc.fovs_order[k]
                if fov_name not in self.track_pt_qc.track_result.keys():
                    continue
                line_fovs[fov_name] = [self.fovs[fov_name], self.track_pt_qc.track_result.get(fov_name)[1]]

            if len(line_fovs) == 0:
                return 1

        track_ln_pc.line_detect(line_fovs, buffer)
        track_ln_pc.track_match()

        if track_ln_pc.score == 0:
            return 1

        self.line_result = track_ln_pc.matching_result
        self.line_score = track_ln_pc.score
        self.best_match = track_ln_pc.get_best_match()

        return 0
