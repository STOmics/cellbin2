import numpy as np
import cv2

from cellbin2.dnn.onnx_net import OnnxNet
from cellbin2.image.wsi_split import SplitWSI
from cellbin2.utils import clog
from cellbin2.image.augmentation import f_gray2bgr

REPRESENT_INV = {
    'black': 0,
    'blur': 1,
    'good': 2,
    'over_expo': 3,
    'uncertain': -1,
}


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / (np.sum(np.exp(x), axis=1, keepdims=True) + 0.00000001)


def pre_func(img_win, win_shape):
    img_win = f_gray2bgr(img_win)
    img_win = cv2.resize(img_win, dsize=win_shape, interpolation=cv2.INTER_LINEAR)
    img_win = np.rollaxis(img_win, 2)
    return img_win


class OnnxMobileNet3(OnnxNet):
    def __init__(
            self,
            weight_path,
            batch_size,
            conf_thresh=0,
            gpu='-1',
            num_threads=0,
    ):
        super().__init__(weight_path, gpu, num_threads)
        self.conf_thresh = conf_thresh
        self.batch_size = batch_size

        self.img_size = (64, 64)
        self.overlap = 0

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        normalization_shape = (1, 3, 1, 1)
        self.mean = np.array([x * 255 for x in mean]).reshape(normalization_shape)
        self.std = np.array([x * 255 for x in std]).reshape(normalization_shape)

    def predict(self, imgs):
        # preprocess
        imgs = np.concatenate(imgs, axis=0)
        imgs = imgs.reshape((-1, 3, self.img_size[0], self.img_size[1]))
        imgs = np.divide((imgs - self.mean), self.std)
        imgs = imgs.astype(np.float32)

        # inference
        # input_name = self.f_predict.get_inputs()[0].name
        output = self.f_predict(imgs)
        labels = output[0]
        labels_s = softmax(labels)

        # postprocess
        topk_ids = np.argmax(labels_s, axis=1, keepdims=True)
        topk_ks = np.max(labels_s, axis=1, keepdims=True)
        # topk_ids_thresh = np.where(topk_ks >= self.conf_thresh, topk_ids, -1)
        # tmp_count = (topk_ids_thresh != -1).sum()
        # print(f"original -> {len(topk_ids)}, thresh -> {tmp_count}")

        preds = np.hstack((topk_ids, topk_ks))
        return preds

    def inference(self, img: np.ndarray):
        """
        1st step: split image into pieces (64, 64)
        2nd step: classify each piece into category (['black', 'over_exposure', 'blur', 'good'])

        Args:
            img (): image in numpy ndarray format

        Returns:
            count_result: counts of each category
            score: clarity score
            pred_re: pred result
                - shape is ceil(image_height / (64 - _overlap)),  ceil(image_width / (64 - self._overlap), 2)
                - 2 -> 1st: class, 2nd probability

        """
        if not isinstance(img, np.ndarray):
            raise Exception(f"Only accept numpy array as input")

        split_run = SplitWSI(
            img=img,
            win_shape=self.img_size,
            overlap=self.overlap,
            batch_size=self.batch_size,
            need_fun_ret=True,
            need_combine_ret=False,
            func_name='Clarity Eval'
        )

        split_run.f_set_run_fun(self.predict)
        split_run.f_set_pre_fun(pre_func, self.img_size)
        box_lst, pred, dst = split_run.f_split2run()
        pred = np.concatenate(pred)  # concatenate batched result

        # post process
        result_shape = (split_run._y_nums, split_run._x_nums)
        pred_re = pred.reshape((result_shape[0], result_shape[1], -1))
        unique, counts = np.unique(pred_re[:, :, 0], return_counts=True)
        unique = unique.astype('int')
        count_result = dict(zip(unique, counts))
        # prob_result = {key: (pred_re[:, :, 1][pred_re[:, :, 0] == key]).sum() for key in unique}
        # good_sum = prob_result.get(REPRESENT_INV['good'], 0)
        # blur_sum = prob_result.get(REPRESENT_INV['blur'], 0)
        # expo_sum = prob_result.get(REPRESENT_INV['over_expo'], 0)
        # if good_sum + blur_sum + expo_sum == 0:
        #     score = 0
        # else:
        #     score = good_sum / (good_sum + blur_sum + expo_sum)

        return count_result, pred_re, box_lst

    @staticmethod
    def score_calculator(pred_arr, weight_map):
        uniques, counts = np.unique(pred_arr[:, :, 0], return_counts=True)
        unique = uniques.astype('int')
        count_result = dict(zip(unique, counts))
        score = 0
        total = 0
        for i in range(len(weight_map)):
            cur_count = count_result.get(i, 0)
            cur_score = cur_count * weight_map.get(i, 0)
            if i != 0:
                total += cur_count
            # print(REPRESENT.get(i, ''), cur_score)
            score += cur_score
        if total == 0:
            final_score = 0
        else:
            final_score = score / total
        count_pct_result = {key: round(item / total, 2) * 100 for key, item in count_result.items()}
        clog.info(f"Clarity evaluation score percentage result: {count_pct_result}")
        return final_score


if __name__ == '__main__':
    pass
