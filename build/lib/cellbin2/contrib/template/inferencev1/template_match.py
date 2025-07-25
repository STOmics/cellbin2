from copy import copy
import math
import numpy as np

from cellbin2.contrib.template.inferencev1.line_detector import Line
from cellbin2.utils import clog


def intersect(s1, s2, s_idx, template, tolerance=15):
    # Calculate the degree of matching
    o_itcp = []
    o_idx = []
    tp_tmp = [template[idx] for idx in s_idx]
    i = j = 0
    st = 0
    ed = 0
    err_dist = 0
    while i < len(s1) and j < len(s2):
        if abs(s1[i] - s2[j]) <= tolerance:
            err_dist += abs(s1[i] - s2[j])
            # If the actual intercept and theoretical intercept are considered to match within the tolerance range
            if len(o_itcp) == 0:
                st = i
            o_itcp.append(s2[j])
            o_idx.append(s_idx[i])
            i += 1
            j += 1
            ed = i
        elif s1[i] - s2[j] < 0:
            i += 1
        else:
            # Set the index to -1 for values that cannot be matched in the actual intercept
            o_idx.append(-1)
            j += 1

    if len(o_idx) < len(s2):
        for k in range(len(s2) - len(o_idx)):
            o_idx.append(-1)

    # Calculate the total width of the standard template through an effective index
    if st == ed:
        tp_dist = 0
    else:
        tp_dist = np.sum(tp_tmp[st:ed - 1])

    return o_itcp, o_idx, tp_dist, err_dist


def map_by_intercept(intercept, template, length=2040, scale_min=0.2, scale_max=2.5):
    # Obtain template width based on index
    def get_template(idx):
        if idx < 0 or idx >= len(template):
            return template[idx % len(template)]
        else:
            return template[idx]

    # Calculate the actual width based on the actual intercept
    interval = [(intercept[i + 1] - intercept[i]) for i in range(len(intercept) - 1)]
    # count = len(interval)
    # template_len = len(template)

    # template_used = template * ((math.ceil(count / template_len)) + 2)
    # template_used.extend(template[:count-1])

    index = []
    eff_itcp = []
    tp_dist = 0
    o_score = 0
    err_min = 0
    # Perform enumeration calculation using each actual intercept and each template width
    for i in range(len(interval)):
        val = interval[i]
        itcp = intercept[i]
        if itcp < 0:
            continue
        for j in range(len(template)):
            fake_itcp = [itcp]
            fake_idx = [j]

            # Calculate temporary scale
            scale_tmp = val / template[j]
            if scale_tmp < scale_min or scale_tmp > scale_max:
                continue

            # Calculate the theoretical intercept list and corresponding index list for this temporary scale
            k = j - 1
            st = itcp - get_template(k) * scale_tmp
            while st >= 0:
                fake_itcp.insert(0, st)
                fake_idx.insert(0, k)
                k -= 1
                st -= get_template(k) * scale_tmp

            k = j
            st = itcp + get_template(k) * scale_tmp
            while st < length:
                fake_itcp.append(st)
                fake_idx.append(k + 1)
                k += 1
                st += get_template(k) * scale_tmp

            for n in range(len(fake_idx)):
                fake_idx[n] = fake_idx[n] % len(template)

            # Calculate and determine the degree of matching
            o_itcp, o_idx, tmp_dist, err_dist = intersect(fake_itcp, intercept, fake_idx, template)
            if len(o_itcp) > o_score:
                o_score = len(o_itcp)
                index = o_idx
                eff_itcp = o_itcp
                tp_dist = tmp_dist
                err_min = err_dist
            elif len(o_itcp) == o_score:
                if err_dist < err_min:
                    o_score = len(o_itcp)
                    index = o_idx
                    eff_itcp = o_itcp
                    tp_dist = tmp_dist
                    err_min = err_dist

    # Result filtering and exception handling
    if o_score <= 2:
        return -1, -1, -1, -1
    if tp_dist == 0:
        return -1, -1, -1, -1

    # Calculate scale
    im_dist = eff_itcp[-1] - eff_itcp[0]

    scale = im_dist / tp_dist

    return index, scale, o_score, err_min


def template_reassign(index, template, scale):
    tr = []
    for i in range(len(index)):
        if index[i] == -1:
            continue
        dist = template[index[i]]
        try:
            dd = [k >= 0 for k in index[i + 1:]].index(True)
        except:
            dd = -1
        if dd >= 0:
            j = index[i] + 1
            if j >= len(template):
                j = j % len(template)
            while j != index[i + dd + 1]:
                dist += template[j]
                j += 1
                if j >= len(template):
                    j = j % len(template)
        tr.append(dist * scale)
    return tr


def intercept_reassign(x0, intercept_):
    intercept = [x0]
    accumulator = x0
    for i in intercept_:
        accumulator += i
        intercept.append(accumulator)
    return intercept


# TODO: template / cos(angle)???
# TODO: template reassign ?????
class TemplateMatcher(object):
    def __init__(self, magnification, scale_range):
        """

        Args:
            scope_type (): 10x or 20x
            scale_range (): scale search range
        """
        if magnification == 10:
            self.scale_min = 1 - scale_range
            self.scale_max = 1 + scale_range
        elif magnification == 20:
            self.scale_min = 2 - scale_range
            self.scale_max = 2 + scale_range
        else:
            raise Exception(f"{magnification} not supported")
        clog.info(f"Using scale min: {self.scale_min}, scale max: {self.scale_max}")
        self.cross_pts = []
        self.x_scale = -1
        self.y_scale = -1
        self.rotation = None
        self.x_cnt = -1
        self.y_cnt = -1
        self.x_err_min = -1
        self.y_err_min = -1

    def template_match(self, x_intercept, y_intercept, shape, chip_template):
        result = [-1] * 10

        x_template = chip_template[0]
        y_template = chip_template[1]

        if len(x_intercept) == 0 or len(y_intercept) == 0:
            return result

        x_index, x_scale, x_cnt, x_err_min = map_by_intercept(
            x_intercept,
            x_template,
            shape[1],
            scale_min=self.scale_min,
            scale_max=self.scale_max
        )
        y_index, y_scale, y_cnt, y_err_min = map_by_intercept(
            y_intercept,
            y_template,
            shape[0],
            scale_min=self.scale_min,
            scale_max=self.scale_max
        )

        if x_scale < 0 or y_scale < 0 or abs(x_scale / y_scale - 1) > 0.1 or len(x_index) < 2 or len(y_index) < 2:
            return result

        if len(y_index) < len(x_index):
            y_scale += (1 - len(y_index) / len(x_index)) * abs(x_scale - y_scale)
        elif len(y_index) > len(x_index):
            x_scale += (1 - len(x_index) / len(y_index)) * abs(x_scale - y_scale)

        x_i = template_reassign(x_index, x_template, x_scale)
        y_i = template_reassign(y_index, y_template, y_scale)

        x_intercept = [x_intercept[x_] for x_ in range(len(x_intercept)) if x_index[x_] >= 0]
        y_intercept = [y_intercept[y_] for y_ in range(len(y_intercept)) if y_index[y_] >= 0]

        x_intercept_ = intercept_reassign(x_intercept[0], x_i[:-1])
        y_intercept_ = intercept_reassign(y_intercept[0], y_i[:-1])

        result = [
            x_intercept_, y_intercept_, x_index, y_index, x_scale, y_scale, x_cnt, y_cnt, x_err_min, y_err_min
        ]

        return result

    def match(
            self,
            shape,
            track_lines,
            chip_template
    ):
        """
        What this function do?
        Given the track line of the image, this algo will match all possible track lines. This algo will get scale and
        rotation based on the possible track lines. After that, this algo will regenerate x and y intercept based on the
        first line in each direction and the corresponding scale.

        Args:
            shape (): shape of the image
            track_lines (): track line of the image
            chip_template ():  the template of the chip

        Results:
            self.cross_pts: list of list (success) -> [[x, y, x_ind, y_ind], ...], None (fail)
            self.x_scale: float (success), None (fail)
            self.y_scale: float (success), None (fail)
            self.rotation: float (success), None (fail)

        Return:
            0: success
            1: fail

        """
        self.cross_pts = []
        self.x_scale = -1
        self.y_scale = -1
        self.rotation = None
        self.x_cnt = -1
        self.y_cnt = -1
        self.x_err_min = -1
        self.y_err_min = -1

        k = list()
        x_intercept = list()
        y_intercept = list()
        counter = 0
        for line in track_lines:
            if abs(line.coefficient) <= 1:
                k.append(line.coefficient)
                y_intercept.append([line.bias, counter])
                counter += 1
            else:
                k.append(-1 / line.coefficient)
                x_intercept.append([line.get_point_by_y(0)[0], counter])
                counter += 1
        x_intercept.sort()
        y_intercept.sort()

        x_intercept_, y_intercept_, x_index, y_index, x_scale, y_scale, x_cnt, y_cnt, x_err_min, y_err_min = \
            self.template_match(
                [val[0] for val in x_intercept],
                [val[0] for val in y_intercept],
                shape,
                chip_template
        )

        if x_intercept_ == -1:
            return 1

        templx_lines = list()
        temply_lines = list()
        line = Line()

        strip = []
        for i in range(len(x_index)):
            if x_index[i] == -1:
                strip.append(x_intercept[i][1])
        for j in range(len(y_index)):
            if y_index[j] == -1:
                strip.append(y_intercept[j][1])

        k = [k[i] for i in range(len(k)) if i not in strip]

        coeff = np.mean(k)
        rotation = math.degrees(math.atan(coeff))

        x_scale = x_scale * math.cos(math.radians(rotation))
        y_scale = y_scale * math.cos(math.radians(rotation))

        x_index = [x for x in x_index if x >= 0]
        y_index = [y for y in y_index if y >= 0]

        for i in range(len(x_intercept_)):
            x = x_intercept_[i]
            line.init_by_point_k([x, 0], -1 / coeff)
            line.index = x_index[i]
            templx_lines.append(copy(line))
        for j in range(len(y_intercept_)):
            y = y_intercept_[j]
            line.init_by_point_k([0, y], coeff)
            line.index = y_index[j]
            temply_lines.append(copy(line))
        templ_points = self.make_cross_points(
            templx_lines=templx_lines,
            temply_lines=temply_lines
        )
        cross_pts = self.point_spread_into_template(
            templ_points=templ_points,
            rotation=rotation,
            shape=shape,
            chip_template=chip_template,
            x_scale=x_scale,
            y_scale=y_scale
        )

        self.cross_pts = cross_pts
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.rotation = rotation
        self.x_cnt = x_cnt
        self.y_cnt = y_cnt
        self.x_err_min = x_err_min
        self.y_err_min = y_err_min

        return 0

    def point_spread_into_template(
            self,
            templ_points,
            rotation,
            shape,
            chip_template,
            x_scale,
            y_scale,
    ):
        x, y, ind_x, ind_y = templ_points[0]
        k0 = math.tan(math.radians(rotation))
        if k0 == 0:
            k0 = 0.00000001
        k1 = -1 / k0
        x0, y0 = 0, 0
        x0 += x
        y0 += y

        y_intercept0 = y0 - k0 * x0
        x_intercept0 = (y0 - k1 * x0) * k0

        dy = abs(k0 * shape[1])
        y_region = (-dy, shape[0] + dy)
        dx = abs(k0 * shape[0])
        x_region = (-dx, shape[1] + dx)
        y_intercept = self.get_intercept(y_intercept0, y_region, ind_y, chip_template[1], y_scale, rotation)
        x_intercept = self.get_intercept(x_intercept0, x_region, ind_x, chip_template[0], x_scale, rotation)
        cross_pts = self.create_cross_points(k0, x_intercept, y_intercept, shape)
        return cross_pts

    def get_intercept(self, intercept0, region, ind, templ, scale, rotation):
        item_count = len(templ)
        idx = intercept0
        intercept = [[idx, ind]]
        s, e = region
        # face to large
        while idx < e:
            ind = ind % item_count
            item_len = (templ[ind] * scale) / math.cos(math.radians(rotation))
            idx += item_len
            intercept.append([idx, (ind + 1) % item_count])
            ind += 1
        # face to small
        idx, ind = intercept[0]
        while idx > s:
            ind -= 1
            ind = ind % item_count
            item_len = (templ[ind] * scale) / math.cos(math.radians(rotation))
            idx -= item_len
            intercept.append([idx, ind])
        return sorted(intercept, key=(lambda x: x[0]))

    def create_cross_points(self, k, x_intercept, y_intercept, shape):
        cross_points = list()
        for x_ in x_intercept:
            for y_ in y_intercept:
                x, ind_x = x_
                y, ind_y = y_
                x0 = (x - k * y) / (pow(k, 2) + 1)
                y0 = k * x0 + y
                if x0 < 0 or x0 > shape[1] or y0 < 0 or y0 > shape[0]:
                    continue
                cross_points.append([x0, y0, ind_x, ind_y])
        return cross_points

    def make_cross_points(self, templx_lines, temply_lines):
        templ_points = list()
        for yl in temply_lines:
            for xl in templx_lines:
                index_y = yl.index
                index_x = xl.index
                y = yl.bias
                k = yl.coefficient
                x = xl.get_point_by_y(0)[0]
                x0 = (x - k * y) / (pow(k, 2) + 1)
                y0 = k * x0 + y
                templ_points.append([x0, y0, index_x, index_y])
        return templ_points


def main():
    import cv2
    from cellbin2.contrib.template.inferencev1.line_detector import TrackLineDetector
    image_path = r"D:\Data\tmp\Y00035MD\Y00035MD\Y00035MD_0000_0005_2023-01-30_15-50-42-418.tif"
    # image_path = r"D:\Data\tmp\Y00035MD\Y00035MD\Y00035MD_0002_0005_2023-01-30_15-50-52-553.tif"
    arr = cv2.imread(image_path, -1)
    # arr = cv.medianBlur(arr, 3)
    print(arr.shape)
    ftl = TrackLineDetector()
    output_path = None
    angle = None
    image_name = None
    h_lines, v_lines = ftl.generate(
        arr=arr,
    )
    track_lines = h_lines + v_lines
    tm = TemplateMatcher()
    chip_template = [[240, 300, 330, 390, 390, 330, 300, 240, 420],
                     [240, 300, 330, 390, 390, 330, 300, 240, 420]]
    ret = tm.match(
        shape=arr.shape,
        track_lines=track_lines,
        chip_template=chip_template
    )

    print(ret)


if __name__ == '__main__':
    main()
