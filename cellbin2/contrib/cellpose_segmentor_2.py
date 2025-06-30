import os
import cv2
import numpy as np
from math import ceil
import patchify
import tifffile
from scipy.ndimage import distance_transform_edt

from cellpose import models, utils
from cellbin2.contrib.cell_segmentor import CellSegParam
from cellbin2.contrib.cellpose_segmentor import f_instance2semantics_max


class CellSegmentation:
    def __init__(self, input_path, save_path=None, photo_size=2048, photo_step=2000, dmin=10, dmax=40, step=10):
        self.input_path = input_path
        self.save_path = save_path
        self.photo_size = photo_size
        self.photo_step = photo_step
        self.dmin = dmin
        self.dmax = dmax
        self.step = step

    def _process_image(self, img_data):
        overlap = self.photo_size - self.photo_step
        if (overlap % 2) == 1:
            overlap = overlap + 1
        act_step = ceil(overlap / 2)
        im = cv2.imread(self.input_path)
        dir_image1 = self.input_path.split('/')[-1].strip('.tif')
        print(dir_image1)
        image = np.array(im)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        res_image = np.pad(gray_image, ((act_step, act_step), (act_step, act_step)), 'constant')
        a = res_image.shape[0]
        b = res_image.shape[1]
        res_a = ceil((a - self.photo_size) / self.photo_step) * self.photo_step + self.photo_size
        res_b = ceil((b - self.photo_size) / self.photo_step) * self.photo_step + self.photo_size
        padding_rows = res_a - a
        padding_cols = res_b - b
        regray_image = np.pad(res_image, ((0, padding_rows), (0, padding_cols)), mode='constant')

        patches = patchify.patchify(regray_image, (self.photo_size, self.photo_size), step=self.photo_step)

        wid = patches.shape[0]
        high = patches.shape[1]
        model = models.Cellpose(gpu=True, model_type='cyto2')
        a_patches = np.full((wid, high, (self.photo_step), (self.photo_step)), 255)

        for i in range(wid):
            for j in range(high):
                img_data = patches[i, j, :, :]
                num0min = wid * high * 800000000000000
                for k in range(self.dmin, self.dmax, self.step):
                    masks, flows, styles, diams = model.eval(img_data, diameter=k, channels=[0, 0], flow_threshold=0.9)
                    masks = f_instance2semantics_max(masks)
                    num0 = np.sum(masks == 0)
                    if num0 < num0min:
                        num0min = num0
                        outlines = utils.masks_to_outlines(masks)
                        outlines = (outlines == True).astype(int) * 255
                        try:
                            a_patches[i, j, :, :] = outlines[act_step:(self.photo_step + act_step),
                                                    act_step:(self.photo_step + act_step)]
                            output = masks.copy()
                        except:
                            a_patches[i, j, :, :] = output[act_step:(self.photo_step + act_step),
                                                    act_step:(self.photo_step + act_step)]
        patch_nor = patchify.unpatchify(a_patches, ((wid) * (self.photo_step), (high) * (self.photo_step)))
        nor_imgdata = np.array(patch_nor)
        cropped_1 = nor_imgdata[0:gray_image.shape[0], 0:gray_image.shape[1]]
        cropped_1 = np.uint8(cropped_1)
        # image_8bit = cv2.convertScaleAbs(cropped_1)
        # inverted_image = cv2.bitwise_not(image_8bit)
        # print(inverted_image.shape)
        return cropped_1

    def _post_image(self, process_image):
        contour_thickness = 0
        contour_coords = np.argwhere(process_image == 255)
        distance_transform = distance_transform_edt(process_image == 0)
        expanded_image = np.zeros_like(process_image)
        for y, x in contour_coords:
            mask = distance_transform[y, x] <= contour_thickness
            expanded_image[y - contour_thickness:y + contour_thickness + 1,
            x - contour_thickness:x + contour_thickness + 1] = mask * 255
        # cv2.imwrite(r"E:\HE_data\img_crop\HE_train_A02085D1_x_5214_y_8103_result2.tif", expanded_image)
        contours, _ = cv2.findContours(expanded_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        print(len(contours))
        height, width = process_image.shape
        black_background = np.zeros((height, width), dtype=np.uint8)
        for contour in contours:
            area = cv2.contourArea(contour)
            if 20 < area < 10000:
                cv2.drawContours(black_background, [contour], -1, 255, thickness=cv2.FILLED)
        black_background = np.uint8(black_background)
        return black_background, expanded_image

    def _merger_image(self, merger_image1, merger_image2):
        merger_image1[merger_image2 == 255] = 0
        return merger_image1

    def segment_cells(self):
        inverted_image = self._process_image(self.input_path)
        # cv2.imwrite(self.save_path, inverted_image)
        post_image, expanded_image = self._post_image(inverted_image)
        print(post_image.dtype)
        # cv2.imwrite(self.save_path, post_image)
        result_image = self._merger_image(post_image, expanded_image)
        if self.save_path is not None:
            cv2.imwrite(self.save_path, result_image)
        return result_image


def segment4cell(input_path: str, cfg: CellSegParam):
    cell_seg = CellSegmentation(
        input_path,
        photo_size=2048,
        photo_step=2000,
        dmin=20,
        dmax=60,
        step=10
    )
    result_image = cell_seg.segment_cells()

    return result_image


if __name__ == '__main__':
    open_path = r"E:\HE_data\img_crop\HE_train_A02085D1_x_5214_y_8103.tif"
    save_path = r"E:\HE_data\img_crop\HE_train_A02085D1_x_5214_y_8103_result.tif"
    photo_size = 512
    photo_step = 500
    dmin = 10
    dmax = 30
    step = 10

    cell_segmenter = CellSegmentation(open_path, save_path, photo_size, photo_step, dmin, dmax, step)
    cell_segmenter.segment_cells()
