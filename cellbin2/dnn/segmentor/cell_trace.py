import cv2
import numpy as np

from cellbin2.image.augmentation import f_rgb2gray
from cellbin2.image.wsi_split import SplitWSI
from cellbin2.utils.pro_monitor import process_decorator


def get_trace(mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    h, w = mask.shape[: 2]
    output = []
    for i in range(num_labels):
        box_w, box_h, area = stats[i][2:]
        if box_h == h and box_w == w:
            continue
        output.append([box_h, box_w, area])
    return output


def get_trace_v2(mask):
    """
    2023/09/20 @fxzhao get_trace的升级版本,分块处理,降低大数据量下的内存占用
    2023/09/21 @fxzhao 增加数据量检测,当数据较小时使用不分块方法
    """
    h, w = mask.shape[: 2]
    steps = 10000
    overlap = 1000
    if h < steps + overlap:
        return get_trace(mask)

    starts = np.array(range(0, h, steps))[:-1]
    starts -= overlap
    ends = starts + steps + overlap * 2
    starts[0] = 0
    ends[-1] = h
    output = []
    for start, end in zip(starts, ends):
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask[start:end, ], connectivity=8)
        up_thre = overlap
        if start == 0:
            up_thre = start
        down_thre = up_thre + steps
        if end == h:
            down_thre = end - start
        for i in range(num_labels):
            _, box_y, box_w, box_h, area = stats[i]
            if box_y < up_thre or down_thre <= box_y:
                continue
            if box_h == (end - start) and box_w == w:
                continue
            output.append([box_h, box_w, area])
    return output


def cal_area(cell_mask, tissue_mask):
    cell_mask = cell_mask * tissue_mask
    area_ratio = np.sum(cell_mask) / np.sum(tissue_mask)
    return area_ratio


def cal_int(c_mask, t_mask, register_img):
    c_mask = c_mask * t_mask
    register_img = f_rgb2gray(register_img, need_not=True)
    c_mask_int = c_mask * register_img
    t_mask_int = t_mask * register_img
    int_ratio = np.sum(c_mask_int) / np.sum(t_mask_int)
    return int_ratio


def get_partial_res(c_mask, t_mask, register_img, keep=5, k=1024):
    def mask_to_outline(mask, img):
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cnts, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, cnts, -1, (255, 0, 0), 1)
        return img

    c_mask = c_mask * t_mask
    sp = SplitWSI(t_mask, win_shape=(k, k))
    sp._f_split()
    boxes = sp.box_lst
    res = []
    for box in boxes:
        y_begin, y_end, x_begin, x_end = box
        sub_mask = c_mask[y_begin: y_end, x_begin: x_end]
        cnts, _ = cv2.findContours(sub_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        counts = len(cnts)
        res.append([box, counts])
    sort_by_counts = sorted(res, key=lambda x: x[1], reverse=True)
    im_outlines = []
    for i in range(keep):
        cur_box = sort_by_counts[i][0]
        y_begin, y_end, x_begin, x_end = cur_box
        cur_box = tuple(cur_box)
        sub_mask = c_mask[y_begin: y_end, x_begin: x_end]
        sub_img = register_img[y_begin: y_end, x_begin: x_end]
        img_with_outline = mask_to_outline(mask=sub_mask, img=sub_img)
        im_outlines.append(tuple((img_with_outline, cur_box)))
    return im_outlines


def cell_int_hist(c_mask, register_img, ifshow=False):
    import matplotlib.pyplot as plt
    from skimage.measure import regionprops
    from scipy import ndimage
    fig, ax = plt.subplots(figsize=(8, 6))
    labeled_img, labels = ndimage.label(c_mask)
    register_img = f_rgb2gray(register_img, need_not=True)
    regions = regionprops(labeled_img, intensity_image=register_img)
    res = []
    for region in regions:
        mean_int = region.intensity_mean
        res.append(mean_int)
    ax.hist(res, bins='auto', density=True, facecolor="tab:blue", edgecolor="tab:orange")
    plt.title("Cell mask intensity histogram")
    plt.xlabel("Pixel value")
    plt.ylabel("Density")
    if ifshow:
        plt.show()
    return fig

def test_hist(cell_mask):
    data = get_trace(cell_mask)

    data = np.array(data)
    d = data[:, 2]
    cellarea_1600 = np.sum(d > 1600) / d.size
    d = data[:, 0]
    cell_height_40 = np.sum(d > 40) / d.size
    d = data[:, 1]
    cell_width_40 = np.sum(d > 40) / d.size

    return cellarea_1600, cell_height_40, cell_width_40


def check_cells_with_tissue(tissue_mask, cell_mask, k):
    def check_area(img):
        h, w = img.shape[:2]
        return int(h * w) == np.sum(img > 0)

    tissue_mask[tissue_mask > 0] = 1
    cell_mask[cell_mask > 0] = 1

    flag = True
    tissue_area = np.sum(tissue_mask > 0)

    cell_mask = cv2.bitwise_and(cell_mask, cell_mask, mask=tissue_mask)
    tmp = np.subtract(tissue_mask, cell_mask)
    sp_run = SplitWSI(tmp, win_shape=(k, k), overlap=0, batch_size=1,
                      need_fun_ret=True, need_combine_ret=False, editable=False, tar_dtype=np.uint8)
    sp_run.f_set_run_fun(check_area)
    _, ret, _ = sp_run.f_split2run()
    ret = np.array(ret).flatten()
    count = np.sum(ret > 0)
    flag = count == 0
    cell_miss_area = count * np.square(k)
    return flag, cell_miss_area / tissue_area


if __name__ == '__main__':
    import sys
    import tifffile
    from cellbin2.image.augmentation import f_ij_16_to_8_v2

    # tissue = tifffile.imread(r"D:\stock\dataset\test\out\FP200000340BR_A1_t.tif")
    regist = tifffile.imread("/media/Data/dzh/data/tmp/test_cseg_report/A01386A4_DAPI_regist.tif")
    c_mask = tifffile.imread("/media/Data/dzh/data/tmp/test_cseg_report/A01386A4_DAPI_mask.tif")
    regist = f_ij_16_to_8_v2(regist)
    # ret = check_cells_with_tissue(tissue, cells, 256)
    # print(ret)

    print(cell_int_hist(c_mask=c_mask, register_img=regist))
    # sys.exit()
