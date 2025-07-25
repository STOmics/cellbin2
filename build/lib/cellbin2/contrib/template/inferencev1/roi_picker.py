import os
import numpy as np
import tifffile as tif
# from cellbin2.utils import rc_key


class TrackROIPicker(object):
    """
    Return images that can be used for template derivation at 20x or 40x
    20x - Return three FOV images
    40x - Return seven FOV images
    """

    def __init__(self, images_path,
                 jitter_list: list() = None,
                 tissue_mask: np.ndarray = None,
                 track_points: dict() = None):

        self.__init_value = 999
        self.rows = None
        self.cols = None

        self.setImagesPath(images_path)
        self.setJitter(jitter_list)
        self.setTissueMask(tissue_mask)
        self.setTrackPoints(track_points)
        self.images_pair = list()

    def setImagesPath(self, images_path):
        """
        :param images_path:
        """
        if isinstance(images_path, dict):
            self.images_path = images_path
        elif isinstance(images_path, str):
            self.images_path = imagespath2dict(images_path)

    def setTissueMask(self, tissue_mask):
        """
        :param tissue_mask:
        """
        assert tissue_mask.shape[0] == self.rows and \
               tissue_mask.shape[1] == self.cols, "Mask shape error."
        self.tissue_mask = tissue_mask

    def setTrackPoints(self, track_points):
        """
        :param track_points: {'r_c': [x, y]}
        """
        self.track_points = track_points

    def setJitter(self, jitter_list, confi_mask=None):
        """
        :param jitter_list: [h_j, v_j]
        """
        h_j, v_j = jitter_list
        assert h_j.shape == v_j.shape, "Jitter ndim is diffient."
        h_j[np.where(h_j == -self.__init_value)] = self.__init_value
        v_j[np.where(v_j == -self.__init_value)] = self.__init_value
        self.horizontal_jitter = h_j
        self.vertical_jitter = v_j
        self.confi_mask = confi_mask

        self.rows, self.cols = self.horizontal_jitter.shape[:2]

    def getRoiImages(self, k=5, threold=None):
        """
        :param k:
        :param threold:
        Return partially stitched image pairs
        """
        all_dst_list = list()
        for row in range(self.rows):
            for col in range(self.cols):
                if self.tissue_mask[row, col] == 0:
                    dst_list = self.__get_jitter(row, col)
                    all_dst_list.extend(dst_list)

        all_dst_list = sorted(all_dst_list, key=self.__get_score)[::-1]

        images_pair = dict()
        for dst_list in all_dst_list[:k]:
            src = dst_list[0]
            src_image = tif.imread(self.images_path[self.__list2str(src)])
            _h, _w = src_image.shape[:2]
            h_coord = list([[0, 0]])
            v_coord = list([[0, 0]])

            for dst in dst_list[1:]:
                dst_image = tif.imread(self.images_path[self.__list2str(dst)])
                if src[1] > dst[1]:
                    x, y = self.horizontal_jitter[src[0], src[1]]
                    h_coord.append([-(_w + x), -y])
                    h_coord = np.array(h_coord)
                    h_image = self.__set_image(h_coord, _h, _w, src_image, dst_image)
                elif src[1] < dst[1]:
                    x, y = self.horizontal_jitter[dst[0], dst[1]]
                    h_coord.append([_w + x, y])
                    h_coord = np.array(h_coord)
                    h_image = self.__set_image(h_coord, _h, _w, src_image, dst_image)
                elif src[0] > dst[0]:
                    x, y = self.vertical_jitter[src[0], src[1]]
                    v_coord.append([-x, -(_h + y)])
                    v_coord = np.array(v_coord)
                    v_image = self.__set_image(v_coord, _h, _w, src_image, dst_image)
                elif src[0] < dst[0]:
                    x, y = self.vertical_jitter[dst[0], dst[1]]
                    v_coord.append([x, _h + y])
                    v_coord = np.array(v_coord)
                    v_image = self.__set_image(v_coord, _h, _w, src_image, dst_image)

            name = ' '.join([rc_key(i[0], i[1]) for i in dst_list])
            images_pair[name] = [h_image, v_image]
        return images_pair

    def __set_image(self, coord, h, w, src, dst):
        x0 = np.min(coord[:, 0])
        y0 = np.min(coord[:, 1])
        x1 = np.max(coord[:, 0])
        y1 = np.max(coord[:, 1])
        coord[:, 0] -= x0
        coord[:, 1] -= y0
        _w = x1 - x0 + w
        _h = y1 - y0 + h

        _image = np.zeros([int(_h), int(_w)], src.dtype)
        for index, image in zip(coord, [src, dst]):
            x, y = index
            _image[y: y + h, x: x + w] = image

        return _image

    def __list2str(self, dst):
        return "_".join([str(i) for i in dst])

    def __get_score(self, dst_list):
        points_num = list()

        d0 = "_".join([str(i) for i in dst_list[0]])
        d1 = "_".join([str(i) for i in dst_list[1]])
        d2 = "_".join([str(i) for i in dst_list[2]])

        for d in [d0, d1, d2]:
            if d in self.track_points.keys():
                points_num.append(len(self.track_points[d]))
            else:
                points_num.append(0)

        return np.mean(points_num) + np.mean(points_num) / (np.std(points_num) + 1)

    def __get_jitter(self, row, col):
        h_list = list()
        v_list = list()
        dst_list = list()

        for index in [col, col + 1]:
            if index < self.cols:
                if self.horizontal_jitter[row, index, 0] != 999:
                    if index == col:
                        if self.tissue_mask[row, index - 1] == 0:
                            h_list.append([row, index - 1])
                    else:
                        if self.tissue_mask[row, index] == 0:
                            h_list.append([row, index])

        for index in [row, row + 1]:
            if index < self.rows:
                if self.vertical_jitter[index, col, 0] != 999:
                    if index == row:
                        if self.tissue_mask[index - 1, col] == 0:
                            v_list.append([index - 1, col])
                    else:
                        if self.tissue_mask[index, col] == 0:
                            v_list.append([index, col])

        for h in h_list:
            for v in v_list:
                dst_list.append([[row, col], h, v])

        return dst_list


if __name__ == '__main__':
    import h5py
    from cellbin2.utils.file_manager import imagespath2dict

    src = r'D:\AllData\Big\2023-02-27\D01653A6B6\D01653A6B6'
    src_fovs = imagespath2dict(src)

    ipr_path = r'D:\AllData\Big\QC\D01653A6B6\D01653A6B6_20230227_175744_1.1.ipr'
    pts = {}
    with h5py.File(ipr_path) as conf:
        qc_pts = conf['QCInfo/CrossPoints/']
        for i in qc_pts.keys():
            pts[i] = conf['QCInfo/CrossPoints/' + i][:]
        h_j = conf['QCInfo']['StitchEval']['HorizontalJitter'][...]
        v_j = conf['QCInfo']['StitchEval']['VerticalJitter'][...]
        confi_mask = conf['QCInfo']['StitchEval']['FOVAlignConfidence'][...]
        tissue_mask = conf['QCInfo']['StitchEval']['FOVTissueType'][...]

    test = TrackROIPicker(images_path=src_fovs, jitter_list=[h_j, v_j], tissue_mask=tissue_mask, track_points=pts)
    images_pair = test.getRoiImages()
