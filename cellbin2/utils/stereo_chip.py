import os
import json
from tkinter import N

import numpy as np
import pandas as pd
from typing import Tuple
from enum import Enum

from cellbin2.utils import clog
from cellbin2.utils.stereo_chip_name import get_chip_prefix_info

try:
    from prettytable import PrettyTable
except ImportError:
    PrettyTable = None


class ChipNameType(Enum):
    LONG = 1
    SHORT = 2
    UNKNOWN = 3


MASK_FOV_LEN = 1470
CHIP_TEMPLATE = [120, 150, 165, 195, 195, 165, 150, 120, 210]
FOV_LEN = 0

# points first coord
POINTS_BEGIN_X = 105
POINTS_BEGIN_Y = 105

#  (163275 - 105) / 1470 = 111
POINTS_END_X = 163275
POINTS_END_Y = 163275

CHIP_TITLE_NAME = ['A', 'B', 'C', 'D', 'Y', 'E', 'F', 'G', 'H', 'Q', 'J', 'K', 'L', 'M', 'N', 'P', 'S']


class StereoChip(object):
    def __init__(self, chip_mask_file: str = '', strong_verification: bool = True):
        """ Coordinate reference
                1   2   3   4   5   6
            G   -   -   -   -   -   -
            F   -   -   -   -   -   -
            E   -   -   -   -   -   -
            D   -   -   -   -   -   -
            C   -   -   -   -   -   -
            A   -   -   -   -   -   -
        """
        if chip_mask_file == '':
            curr_path = os.path.dirname(os.path.realpath(__file__))
            chip_mask_file = os.path.join(curr_path, r'../config/chip_mask.json')
        with open(chip_mask_file, 'r') as fd:
            self.chip_mask = json.load(fd)
        self._name: str = None
        self.chip_specif = ''  # chip specifications:S0.5, S1
        self.fov_edge_len = 2940
        self.fov_template = [[240, 300, 330, 390, 390, 330, 300, 240, 420],
                             [240, 300, 330, 390, 390, 330, 300, 240, 420]]
        self.name_type = ''  # long code or short code
        self.width = 0
        self.height = 0
        self._00pt: Tuple[int, int] = (0, 0)
        self._chip_00pt: Tuple[int, int] = (0, 0)
        self.is_from_S13: bool = False  # seperated from S13 or S6
        self.S1_fov_count = 6.8
        self.exp_r = [0.2, 0.2]  # WH

        self._s13_label = ['Y', 'Q']
        self._chip_rows = '0_0'
        self._chip_cols = '0_0'
        self._template_points: np.ndarray = None

    @staticmethod
    def _create_points(b, e):
        p_list = list([b])

        _p = b
        k = 0
        while _p < e:
            _p += CHIP_TEMPLATE[k % 9]
            p_list.append(_p)
            k += 1

        return p_list

    def create_track_points(self):
        """ To the distribution of points on the S13 large and small chips

        Returns:

        """
        all_points_list = list()

        x_list = self._create_points(POINTS_BEGIN_X, POINTS_END_X)
        y_list = self._create_points(POINTS_BEGIN_Y, POINTS_END_Y)

        for _x in x_list:
            for _y in y_list:
                all_points_list.append([_x, _y])

        return all_points_list

    def set_zero_point_info(self, ):
        """

        Args:
            chip_name:

        Returns:

        """
        track_points_data = self.create_track_points()
        track_points_data = pd.DataFrame(track_points_data, columns = ['x', 'y'])

        _info = None
        if self.is_from_S13:
            if min(self.chip_specif) == 0.5:
                info = self.chip_mask['T10_90_90_s'][self._name[-4:]]
                sx, sy = self._get_start_margin(self.chip_mask['T10_90_90_s'])
            else:
                if max(self.chip_specif) > 1:
                    _info = self.chip_mask['T10_90_90_230508'][self._name[-4:-2]]
                info = self.chip_mask['T10_90_90_230508'][self._name[-2:]]
                sx, sy = self._get_start_margin(self.chip_mask['T10_90_90_230508'])

        else:
            if min(self.chip_specif) == 0.5:
                info = self.chip_mask['T10_41_41_s'][self._name[-4:]]
                sx, sy = self._get_start_margin(self.chip_mask['T10_41_41_s'])
            else:
                if max(self.chip_specif) > 1:
                    _info = self.chip_mask['T10_41_41_230508'][self._name[-4:-2]]
                info = self.chip_mask['T10_41_41_230508'][self._name[-2:]]
                sx, sy = self._get_start_margin(self.chip_mask['T10_41_41_230508'])

        self._chip_rows, self._chip_cols, lr, ud = self._get_chip_rc(info, _info)
        self.exp_r = [round(sum(lr), 1), round(sum(ud), 1)]

        first_xy = list(map(lambda x: x * MASK_FOV_LEN, (sx, sy)))

        row = list(map(int, self._chip_rows.split("-")))
        col = list(map(int, self._chip_cols.split("-")))

        mask_x_min = (col[0] - 1) * MASK_FOV_LEN
        mask_x_max = col[1] * MASK_FOV_LEN
        mask_y_min = (row[0] - 1) * MASK_FOV_LEN
        mask_y_max = row[1] * MASK_FOV_LEN

        fov_count = self.S1_fov_count / 2 if min(self.chip_specif) == 0.5 else self.S1_fov_count
        fov_x_min = round(first_xy[0] + int(col[0] / fov_count) *
                          fov_count * MASK_FOV_LEN)
        fov_x_max = round(first_xy[0] + int(col[1] / fov_count) *
                          fov_count * MASK_FOV_LEN)
        fov_y_min = round(first_xy[1] + int(row[0] / fov_count) *
                          fov_count * MASK_FOV_LEN)
        fov_y_max = round(first_xy[1] + int(row[1] / fov_count) *
                          fov_count * MASK_FOV_LEN)

        points_fov = track_points_data.loc[
            (track_points_data['x'] > fov_x_min) &
            (track_points_data['y'] > fov_y_min) &
            (track_points_data['x'] < fov_x_max) &
            (track_points_data['y'] < fov_y_max)].to_numpy()

        points_finish = (points_fov - [mask_x_min, mask_y_min]) * 2
        # self._template_points = points_finish

        x_set, y_set = map(lambda x: sorted(set(x)),
                           points_finish.transpose(1, 0).tolist())
        _x = np.array(x_set)[1:] - np.array(x_set)[:-1]
        _y = np.array(y_set)[1:] - np.array(y_set)[:-1]

        _x_index = np.where(_x == max(self.fov_template[0]))[0][0]
        _y_index = np.where(_y == max(self.fov_template[0]))[0][0]

        _x_index, _y_index = map(lambda x: -1 if x == 8 else x, (_x_index, _y_index))

        zero_x = x_set[_x_index + 1]
        zero_y = y_set[_y_index + 1]

        n = points_finish.shape[0]
        points_with_indices = np.zeros((n,4))
        points_with_indices[:, :2] = points_finish
        for i in range(n):
            x, y = points_finish[i]
            x_idx = np.where(x_set == x)[0][0]
            y_idx = np.where(y_set == y)[0][0]
            i_index = (x_idx - np.where(np.array(x_set) == zero_x)[0][0]) % 9
            j_index = (y_idx - np.where(np.array(y_set) == zero_y)[0][0]) % 9
            points_with_indices[i, 2] = i_index
            points_with_indices[i, 3] = j_index
        self._template_points = points_with_indices

        # added: distance between 00 point and chip corner
        index = np.where(((points_finish[:, 0] == zero_x) & (points_finish[:, 1] == zero_y)) == True)[0][0]
        chip_point = (points_fov[index] - [fov_x_min, fov_y_min]) * 2

        self._00pt = (zero_x, zero_y)  # distance between 00 point and matrix left top corner coordinate
        self._chip_00pt = chip_point  # distance between 00 point and chip left top corner coordinate

    def _get_chip_rc(self, info1, info2):
        if max(self.chip_specif) > 1:
            row_list = list(map(int, info1["fov_row"].split("-"))) + \
                       list(map(int, info2["fov_row"].split("-")))
            col_list = list(map(int, info1["fov_col"].split("-"))) + \
                       list(map(int, info2["fov_col"].split("-")))

            row = "-".join([str(min(row_list)), str(max(row_list))])
            col = "-".join([str(min(col_list)), str(max(col_list))])

            lr = [info2["lr_expand"][0], info1["lr_expand"][1]]
            ud = [info2["ud_expand"][1], info1["ud_expand"][0]]

        else:
            row = info1["fov_row"]
            col = info1["fov_col"]

            lr = info1["lr_expand"]
            ud = info1["ud_expand"]

        return row, col, lr, ud

    @property
    def template_points(self, ):
        return self._template_points

    @property
    def zero_zero_point(self, ): return self._00pt

    @property
    def zero_zero_chip_point(self, ): return self._chip_00pt

    @property
    def norm_chip_size(self, ):
        w, h = self.chip_specif
        return w * self.S1_fov_count * self.fov_edge_len, h * self.S1_fov_count * self.fov_edge_len

    @staticmethod
    def _get_start_margin(info):

        for k, v in info.items():
            row, col, lr, ud = list(v.items())
            rx, ry = map(int, row[1].split("-"))
            cx, cy = map(int, col[1].split("-"))
            if rx == cx == 1:
                sx, sy = lr[1][0], ud[1][0]
                break

        return sx, sy

    def get_chip_specif_str(self, ):
        return '{}x{}'.format(self.chip_specif[1], self.chip_specif[0])  # letters (rows) first, numbers (columns) second

    def get_version(self, ): return self._name[1:6]

    @property
    def chip_name(self, ):
        return self._name

    def update_expr(self, ):
        # TODO: set by the real rule
        pass

    def set_chip_size(self, ):
        _ = self.S1_fov_count * self.fov_edge_len
        w, h = self.chip_specif
        self.width = int(self.fov_edge_len * (w * self.S1_fov_count + self.exp_r[0]))
        self.height = int(self.fov_edge_len * (h * self.S1_fov_count + self.exp_r[1]))

    def set_chip_specif(self, ):

        def suffix_parser(suffix, name):
            if suffix[-2:] in ['11', '12', '13', '14']:
                specif = [0.5, 0.5]
            else:
                if suffix[-2] not in name or suffix[-4] not in name:
                    h = ord(suffix[-2]) - ord(suffix[-4])
                else:
                    h = name.index(suffix[-2]) - name.index(suffix[-4])

                w = int(suffix[-1]) - int(suffix[-3])
                specif = [w + 1, h + 1]
            return specif

        title_name = sorted(set([i[0] for i in list(self.chip_mask['T10_90_90_230508'].keys())]))
        name_len = len(self._name)
        try:
            if self.name_type == ChipNameType.SHORT:
                if name_len == 8:
                    self.chip_specif = [1, 1]
                else:
                    self.chip_specif = suffix_parser(self._name[-4:], title_name)
            elif self.name_type == ChipNameType.LONG:
                suffix = self._name.split('_')[-1]
                if len(suffix) == 2:
                    self.chip_specif = [1, 1]
                else:
                    self.chip_specif = suffix_parser(suffix, title_name)
            else:
                self.chip_specif = [1, 1]
        except ValueError:
            self.chip_specif = [1, 1]
        except Exception as e:
            clog.warning(
                f'{e}, set chip specif to [1, 1]'
            )
            self.chip_specif = [1, 1]

    def is_after_230508(
            self,
            s13_min_num = 395,
            s6_min_num = 3205,
            deprecated_word = ["B", "I", "O"]
    ) -> bool:
        """
        Registration pre-process uses this parameter as a condition,
        and it requires the chip placement angle Rot90=0 during imaging
        :param s13_min_num:
        :param s6_min_num:
        :param deprecated_word:
        :return: True means chip produced after 05/23/23
        """
        try:
            if self.name_type is ChipNameType.SHORT:

                for _w in deprecated_word:
                    if _w in self.chip_name[-4:]:
                        return False

                    prefix = self.chip_name[0]
                    num_str = self.chip_name[1:6]
                    # determine if SN[1:6] consists entirely of numbers
                    if num_str.isdigit():
                        num_val = int(num_str)
                        if prefix in ['A', 'B', 'C', 'D']:  # 'A00025A1','A00025A111','A03025A1',
                            return num_val >= s6_min_num
                        elif prefix == 'Y':
                            return num_val >= s13_min_num
                        else:
                            # SN[0] support [E, F, G, H, Q, ...] + serial number supports [any 5-digit number], valid
                            return True
                    else:
                        # SN[0] support [E, F, G, H, Q, ...] + any 5 digits and letters, valid
                        return True
            else:  # do not support long code for now
                return False
        except ValueError:
            return False
        except Exception as e:
            clog.warning(f'{e}')
            return False

    def _unknown_process(self):
        clog.warning('Chip name {} lengths is not valid, set chip type to UNKNOWN.'.format(self.chip_name))
        self.name_type = ChipNameType.UNKNOWN

        self.chip_specif = [1, 1]
        self.is_from_S13 = False
        self.exp_r = [0.2, 0.2]
        self._chip_rows = self._chip_cols = '0_0'
        self._00pt = [3150, 3150]
        self._chip_00pt = [0, 0]
        self.set_chip_size()

        # clog.info('Parse chip info as [SN, NameType, fromS13, Specif, Size(WH)] == ({}, {}, {}, {}, {})'.format(
        #     self._name, self.name_type, self.is_from_S13, self.get_chip_specif_str(), (self.width, self.height)))

    def _base_rules(self, ):
        name_len = len(self._name)

        if name_len == 0:
            return 0

        # 首位必须是字母
        if not self._name[0] in CHIP_TITLE_NAME:
            return 0

        if name_len in [8, 10]:
            if not self._name[6].isalpha():
                return 0
            if (name_len == 8) and (self._name[0] not in self._s13_label):
                if not self._name[7].isdigit():
                    return 0
            self.name_type = ChipNameType.SHORT
        elif name_len in [14, 16, 17, 18]:
            self.name_type = ChipNameType.LONG
        else:
            return 0

        return 1

    def _print_info(self, ):
        """

        Returns:

        """
        if PrettyTable is None:
            return

        tb = PrettyTable(["Chip Name", "Chip Type", "S6 | S13", "Chip Size",
                          "Points_00", "Points_chip_00", "Rows", "Cols", "Expand", "Shape"])
        tb.add_row(
            [self._name, self.name_type, 'S13' if self.is_from_S13 else 'S6', self.get_chip_specif_str(),
             self._00pt, self._chip_00pt, self._chip_rows, self._chip_cols, self.exp_r,
             [self.height, self.width]]
        )
        clog.info(f"\n{tb}")

    def parse_info(self, chip_no: str, print_flag = True):
        """
        Args:
            chip_no:
            print_flag:

        Returns:

        """
        self._name = chip_no

        # scn = get_chip_prefix_info(self._name[0], self._name)

        key = self._base_rules()
        if key == 0:
            self._unknown_process()
            if print_flag:
                self._print_info()
            return

        self.is_from_S13 = self._name[0] in self._s13_label and True or False  # determine S13
        self.set_chip_specif()
        if self.is_after_230508():
            try:
                self.set_zero_point_info()
            except KeyError:
                self._unknown_process()
                if print_flag:
                    self._print_info()
                return

        self.set_chip_size()

        if print_flag:
            self._print_info()
        # clog.info('Parse chip info as [SN, NameType, fromS13, Specif, Size(WH)] == ({}, {}, {}, {}, {})'.format(
        #     self._name, self.name_type, self.is_from_S13, self.get_chip_specif_str(), (self.width, self.height)))


def main():
    curr_path = os.path.dirname(os.path.realpath(__file__))
    chips = ['SS200000045_M5',
             'A00009A1', 'B03925A5', 'D03466F1G2', 'Y00063C8D8', 'B04001D211', 'B04001D414',
             'DP8400000006BL_E4', 'DP8400028436TR_A3', 'FP200009158_M1', 'FP200009107_E312',
             'FP200000407BR_F2', 'FP200000561BL_A3B4', 'SS200000672_CC', 'SS200000060_K5L5',
             'SS200000060_M3N3', 'SS200000979TL_E6', 'SS200000108BR_A3A4']
    for chip_name in chips:
        sc = StereoChip(chip_mask_file=os.path.join(curr_path, r'../config/chip_mask.json'))
        sc.parse_info(chip_no=chip_name)


if __name__ == '__main__':
    # main()
    curr_path = os.path.dirname(os.path.realpath(__file__))
    sc = StereoChip(chip_mask_file = os.path.join(curr_path, r'../config/chip_mask.json'))
    sc.parse_info(chip_no = 'Y40320PA', print_flag=True)
    print(1)
    # word = 'ACDEFGHJKLMNP'
    # num = '123456789ACDE'
    #
    # points_list = []
    #
    # for i in range(len(word)):
    #     for j in range(len(num)):
    #         for k in range(4):
    #             sc.parse_info(chip_no='Y03950' + word[i] + num[j] + f'1{k + 1}')
    #             # zz = sc.zero_zero_point
    #             # print(zz)
    #             # print("***************")
    #             # print(np.array(sc.zero_zero_point))
    #             points_list.append(np.array(sc.zero_zero_chip_point))
    #             # print(np.array(sc.zero_zero_chip_point))
    #             # print(2940 - np.array(sc.zero_zero_chip_point))
    #
    # print(1)
    # np.savetxt(r"G:\temp\S13_small.txt", points_list)

