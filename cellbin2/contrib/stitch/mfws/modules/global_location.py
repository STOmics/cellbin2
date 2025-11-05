"""
Solve splicing coordinates
"""
import os
import glog
import numpy as np

from sklearn.linear_model import LinearRegression
import scipy.spatial as spt


class GlobalLocation(object):
    """
    Only receive the offset matrix and solve for the final stitching coordinates
    """
    def __init__(self):
        self.overlap_x = self.overlap_y = 0.1
        self.fov_loc_array = None
        self.offset_diff = None
        self.__init_value = 999
        self.horizontal_jitter = None
        self.vertical_jitter = None
        self.fov_height = None
        self.fov_width = None
        self.rows = None
        self.cols = None

    def set_overlap(self, overlap_x, overlap_y):
        """

        Args:
            overlap_x:
            overlap_y:

        Returns:

        """
        self.overlap_x = overlap_x
        self.overlap_y = overlap_y

    def set_jitter(self, h_j, v_j):
        assert h_j.shape == v_j.shape, "Jitter ndim is diffient."
        h_j[np.where(h_j == -self.__init_value)] = self.__init_value
        v_j[np.where(v_j == -self.__init_value)] = self.__init_value
        self.horizontal_jitter = h_j
        self.vertical_jitter = v_j

    def set_image_shape(self, height, width):
        assert isinstance(height, int) or isinstance(height, float), "Height type error."
        assert isinstance(width, int) or isinstance(width, float), "Width type error."
        self.fov_height = int(height)
        self.fov_width = int(width)

    def set_size(self, rows, cols):
        assert isinstance(rows, int) or isinstance(rows, float), "Rows type error."
        assert isinstance(cols, int) or isinstance(cols, float), "Cols type error."
        self.rows = rows
        self.cols = cols

    def create_location(self, mode='cd'):
        """
        :param mode: Generate final coordinates 'cd'
        """
        if mode == 'cd':
            coord_model = CenterLrDiffuseStitch(self.rows, self.cols)
            coord_model.set_jitter(self.horizontal_jitter, self.vertical_jitter)
            coord_model.set_scope_loc_by_overlap(self.fov_height, self.fov_width, self.overlap_x, self.overlap_y)
            coord_model.multi_connect_domain_center()
            coord_model.multi_center_2_global_loc(center_start=False)

            self.offset_diff = coord_model.offset_diff
            self.fov_loc_array = coord_model.global_loc
        elif 'LS' in mode:
            if mode == 'LS-H':
                lsv = LineScanStitch(self.rows, self.cols, ls=0)
            elif mode == 'LS-V':
                lsv = LineScanStitch(self.rows, self.cols, ls=1)
            else:
                raise ValueError("Stitch method name error.")

            lsv.set_jitter(self.horizontal_jitter, self.vertical_jitter)
            lsv.set_overlap(self.overlap_x, self.overlap_y)
            lsv.set_scope_loc_by_overlap(self.fov_height, self.fov_width)
            new_location = lsv.get_location()
            self.fov_loc_array = new_location

        else:
            pass


##################################
"""Coordinate generation method"""
##################################


class LineScanStitch:
    """ Line scanning splicing, with no overlap or a constant overlap in a certain direction """

    def __init__(self, rows, cols, ls = 0):
        # Line scan direction 0 means horizontal 1 means vertical
        self.line_scan = ls

        self.overlap_x = self.overlap_y = 0.1
        self.scope_global_loc = None
        self.rows = rows
        self.cols = cols
        self.fov_height = None
        self.fov_width = None

        self.horizontal_jitter = None
        self.vertical_jitter = None

        self._placeholder = 999

    def set_jitter(self, h_j, v_j):
        """ set jitter matrix """
        assert h_j.shape == v_j.shape, "jitter ndim is diffient."
        self.horizontal_jitter = h_j
        self.vertical_jitter = v_j

    def set_fov_size(self, fov_height, fov_width):
        """ set fov size """

        self.fov_height = fov_height
        self.fov_width = fov_width

    def set_overlap(self, overlap_x, overlap_y):
        """
        Args:
            overlap_x:
            overlap_y:

        Returns:
        """

        self.overlap_x = overlap_x
        self.overlap_y = overlap_y

    def set_scope_loc_by_overlap(self, h, w):
        """ Calculate the original stitching coordinates of the microscope """

        scope_global_loc = np.zeros(shape=(self.rows, self.cols, 2), dtype=np.int32)
        self.fov_height = h
        self.fov_width = w
        for i in range(self.rows):
            for j in range(self.cols):
                scope_global_loc[i, j, 0] = j * (w - int(w * self.overlap_x))
                scope_global_loc[i, j, 1] = i * (h - int(h * self.overlap_y))
        self.scope_global_loc = scope_global_loc

    def get_offset_by_jitter(self, jit):
        if self.line_scan == 0:
            offset = []
        else:
            offset = np.zeros([jit.shape[1], 2], dtype = np.int_) + self._placeholder
            for c in range(jit.shape[1]):
                jit_x, jit_y = map(
                    lambda x: [i for i in list(jit[:, c, x].ravel()) if i != self._placeholder],
                    (0, 1)
                )

                jit_x, jit_y = map(
                    lambda x: [self._placeholder] if len(x) == 0 else x,
                    (jit_x, jit_y)
                )

                offset[c] = [int(np.median(jit_x)), int(np.median(jit_y))]

            offset_odd = np.round(np.mean(
                [i for i in list(offset[1::2, :]) if i[0] != self._placeholder], axis=0)
            )
            offset_even = np.round(np.mean(
                [i for i in list(offset[::2, :]) if i[0] != self._placeholder], axis=0)
            )
            try:
                len(offset_even)
            except TypeError:
                offset_even = [offset_odd[0], -offset_odd[1]]
            try:
                len(offset_odd)
            except TypeError:
                offset_odd = [offset_even[0], -offset_even[1]]

            for c in range(1, offset.shape[0]):
                if offset[c][0] == self._placeholder or \
                        offset[c][1] == self._placeholder:
                    if c % 2 == 0:
                        offset[c] = offset_even
                    else:
                        offset[c] = offset_odd

        return offset

    def get_location(self):
        new_location = self.scope_global_loc.copy()
        if self.line_scan == 0:
            pass
        else:
            offset = self.get_offset_by_jitter(self.horizontal_jitter)
            for c in range(1, self.cols):
                if c == 1:
                    new_location[:, c] = new_location[:, c] + \
                                         offset[c] + \
                                         [int(self.fov_width * self.overlap_x), 0]
                else:
                    new_location[:, c] = new_location[:, c - 1] + \
                                         offset[c] + \
                                         [self.fov_width, 0]

        return new_location


class CenterLrDiffuseStitch:
    """ Derive the corresponding splicing sequence from the central position of the organizational connected domain
        * * *    * # *    # # #
        * # * -> # # # -> # # #
        * * *    * # *    # # #
    """

    def __init__(self, rows, cols):
        """ rows, cols:
            cumulate_thread: int: cumulative departure by centerLrDiffuseStitch
            src: (row, col)
        """
        self.scope_global_loc = None
        self.rows = rows
        self.cols = cols
        self.fov_height = None
        self.fov_width = None

        self.horizontal_jitter = None
        self.vertical_jitter = None
        self.domain_list = None

        # self.src = src

        self.jitter_mask = np.zeros((self.rows, self.cols), dtype=int) + 999  # Connected domain information
        self.stitch_mask = np.zeros((self.rows, self.cols), dtype=int)  # Index to be spliced
        self.stitch_masked = np.zeros((self.rows, self.cols), dtype=int)  # Spliced index
        self.global_loc = np.zeros((self.rows, self.cols, 2), dtype=int) + 999
        self.stitch_list = list()  # Final splicing sequence
        self.connect_domains = dict()  # Connected domain information
        self.domains = -1  # Connected domain number
        self.offset_diff = np.zeros(shape=(self.rows, self.cols)) - 1  # estimate accumulative error

        # output
        self.output = None
        self.cumulate_offset: int = 999

    def set_jitter(self, h_j, v_j):
        """ set jitter matrix """

        assert h_j.shape == v_j.shape, "jitter ndim is diffient"
        self.horizontal_jitter = h_j
        self.vertical_jitter = v_j

    def set_scope_loc(self, scope_loc: np.ndarray):
        """ set scope loction """
        self.scope_global_loc = scope_loc

    def set_fov_size(self, fov_height, fov_width):
        """set fov size"""
        self.fov_height = fov_height
        self.fov_width = fov_width

    def set_scope_loc_by_overlap(self, h, w, overlap_x, overlap_y):
        """calculate the original stitching coordinates of microscopy """
        scope_global_loc = np.zeros(shape=(self.rows, self.cols, 2), dtype=np.int32)
        self.fov_height = h
        self.fov_width = w
        for i in range(self.rows):
            for j in range(self.cols):
                scope_global_loc[i, j, 0] = j * (w - int(w * overlap_x))
                scope_global_loc[i, j, 1] = i * (h - int(h * overlap_y))
        self.scope_global_loc = scope_global_loc

    def neighbor(self, row, col, mask: np.ndarray = None):
        """
        Gets the unstitching FOV of the specified column position attachment
        """

        if (col - 1) < 0:
            left = None
        else:
            left = (row, col - 1)
        if (col + 1) > (self.cols - 1):
            right = None
        else:
            right = (row, col + 1)
        if (row - 1) < 0:
            up = None
        else:
            up = (row - 1, col)
        if (row + 1) > (self.rows - 1):
            down = None
        else:
            down = (row + 1, col)

        if mask is None:
            return [left, right, up, down]
        else:
            mask_list = list()
            for it in [left, right, up, down]:
                if it and mask[it[0], it[1]] == 1:
                    mask_list.append(it)
            return mask_list

    def neighbor_fix(self, row, col):
        """
        Gets the unstitching FOV of the specified column position attachment
        :return: [r, c]
        """
        if (col - 1) < 0:
            left = None
        else:
            left = [row, col - 1]
        if (col + 1) > (self.cols - 1):
            right = None
        else:
            right = [row, col + 1]
        if (row - 1) < 0:
            up = None
        else:
            up = [row - 1, col]
        if (row + 1) > (self.rows - 1):
            down = None
        else:
            down = [row + 1, col]

        for it in [left, right, up, down]:
            if it:
                if self.stitch_masked[it[0], it[1]] == 1:
                    return it
        return None

    def _getFirstPosition(self, r=True):
        '''get the first position of connect domain
        r: a parameter controls wheather add name for connect domain
        '''
        for i in range(self.rows):
            for j in range(self.cols):
                if self.horizontal_jitter[i, j, 0] != 999 or \
                        self.vertical_jitter[i, j, 0] != 999:
                    if (i, j) not in self.connect_domains.keys():
                        if r:
                            self.domains += 1
                        return (i, j)
        return None

    def _indexIsLegal(self, index):
        """ Determine whether the row and column numbers are legal """

        row, col = index
        if 0 <= row < self.rows and \
                0 <= col < self.cols:
            return True
        return False

    def caculateCenter(self, dst=None):
        """ Recursively find connected regions """
        h_flag = False
        new_dst = list()
        if dst is None:
            dst = [self._getFirstPosition()]

        for (row, col) in dst:
            # if (row, col) not in self.connect_domains.keys():
            self.connect_domains[(row, col)] = self.domains
            self.jitter_mask[row, col] = self.domains
            # temp = self.neighbor(row, col)

            if self.horizontal_jitter[row, col, 0] != 999:
                if self._indexIsLegal((row, col - 1)):
                    if (row, col - 1) not in self.connect_domains.keys():
                        self.connect_domains[(row, col - 1)] = self.domains
                        self.jitter_mask[row, col - 1] = self.domains
                        if (row, col - 1) not in new_dst:
                            new_dst.append((row, col - 1))
            if self.vertical_jitter[row, col, 0] != 999:
                if self._indexIsLegal((row - 1, col)):
                    if (row - 1, col) not in self.connect_domains.keys():
                        self.connect_domains[(row - 1, col)] = self.domains
                        self.jitter_mask[row - 1, col] = self.domains
                        if (row - 1, col) not in new_dst:
                            new_dst.append((row - 1, col))

            if self._indexIsLegal((row, col + 1)):
                if self.horizontal_jitter[row, col + 1, 0] != 999:
                    if (row, col + 1) not in self.connect_domains.keys():
                        if (row, col + 1) not in new_dst:
                            new_dst.append((row, col + 1))
            if self._indexIsLegal((row + 1, col)):
                if self.vertical_jitter[row + 1, col, 0] != 999:
                    if (row + 1, col) not in self.connect_domains.keys():
                        if (row + 1, col) not in new_dst:
                            new_dst.append((row + 1, col))

        if len(new_dst) != 0:
            self.caculateCenter(new_dst)
        else:
            return

    def caculateDomains(self):
        ''' repeatedly search for connected domains until search complete
        return: jitter_mask, connect_domains
        '''
        while self._getFirstPosition(r=False) is not None:
            self.caculateCenter()

    @staticmethod
    def _get_nearest_pts(src_pt, dst_pts: np.ndarray):
        """ find nearst pts
            Args:
                scr_pts:
                dst_pts:
            Returns:
                nearst pts
        """

        tree = spt.cKDTree(data=dst_pts[:, :2])
        distance, index = tree.query(src_pt, k=1)  # find the closest point to template point
        template_point = dst_pts[index]
        return distance, template_point

    def getStitchCenter(self, max_domain=None):
        ''' get the index of stitching center of the max connect domain '''
        if max_domain is None:
            max_domain = max(self.connect_domains.values(), key=list(self.connect_domains.values()).count)
        else:
            assert max_domain in np.arange(0, self.domains + 1), 'max_domain input error'
        domains = np.array([i for i in self.connect_domains.keys() if self.connect_domains[i] == max_domain])
        max_row = np.max(domains[:, 0])
        min_row = np.min(domains[:, 0])
        max_col = np.max(domains[:, 1])
        min_col = np.min(domains[:, 1])
        center_row = int((max_row + min_row) / 2)
        center_col = int((max_col + min_col) / 2)

        center = np.array([center_row, center_col])
        dis, (center_row, center_col) = self._get_nearest_pts(center, domains)

        return center_row, center_col

    '''
        connect to studio
        '''

    def _getStitchOrder_by_domain(self, stitch_order):
        ''' get the order by domain  '''
        domain_mask = np.zeros(shape=(self.rows, self.cols))
        stitch_list = []
        for key, value in self.connect_domains.items():
            if value == self.connect_domains[stitch_order[0]]:
                row, col = key
                domain_mask[row, col] = 1

        while len(stitch_order) > 0:
            index = stitch_order.pop(0)
            if index not in stitch_list:
                stitch_list.append(index)
                nei = self.neighbor(index[0], index[1], mask=domain_mask)
                for i in nei:
                    if i not in stitch_list and i is not None:
                        stitch_order.append(i)
        return stitch_list

    def _getStitchOrder(self, stitch_order):
        '''get the order'''
        # domain_mask = np.zeros(shape=(self.rows,self.cols))
        stitch_list = []

        while len(stitch_order) > 0:
            index = stitch_order.pop(0)
            if index not in stitch_list:
                stitch_list.append(index)
                nei = self.neighbor(index[0], index[1])
                for i in nei:
                    if i not in stitch_list and i is not None:
                        stitch_order.append(i)
        return stitch_list

    def centerToGlobal(self, row=None, col=None):
        '''find the stitch order from center
        support key input: row, col
        otherwise calculate the central of max connected domain
        '''
        if row is None and col is None:
            self.caculateDomains()
            row, col = self.getStitchCenter()
            self.center_row = row
            self.center_col = col

        stitch_order = list()
        stitch_order.append((row, col))
        self.stitch_list = self._getStitchOrder(stitch_order)

        # return stitch_order

    def domain_order(self, max_domain):
        """
        start from specified connected domain, stitch the corresponding connected domain by sequence
        :param max_domain:
        :return:
        """
        first_row, first_col = self.getStitchCenter(max_domain)

        domain_order = [(first_row, first_col)]
        domain_list = []
        domain_pass_list = []
        while len(domain_order) > 0:
            index = domain_order.pop(0)
            if index not in domain_pass_list:
                domain_pass_list.append(index)
                nei = self.neighbor(index[0], index[1])
                for i in nei:
                    if i is not None:
                        domain_order.append(i)
                        domain = self.jitter_mask[i[0], i[1]]
                        if domain not in domain_list and domain != 999:
                            domain_list.append(domain)
        return domain_list

    def multi_connect_domain_center(self):
        '''find the stitch order from central
        support key input: row, col
        otherwise calculate the central of max connected domain
        '''
        # if row is None and col is None:
        self.caculateDomains()
        max_domain = max(self.connect_domains.values(), key=list(self.connect_domains.values()).count)
        self.domain_list = self.domain_order(max_domain)

    def multi_center_2_global_loc(self, center_start=False):
        self.lr = LinearRegression()
        self.lr.fit(np.array([[0, 0]]), np.array([[0, 0]]))
        if (self.lr.coef_ == 0).all():
            self.lr.coef_ = np.diag(np.ones(2))

        for domain_idx in self.domain_list:
            if center_start:
                center_row, center_col = int(self.rows / 2), int(self.cols / 2)
            else:
                center_row, center_col = self.getStitchCenter(max_domain=domain_idx)

            stitch_order = list()
            stitch_order.append((center_row, center_col))
            stitch_list = self._getStitchOrder_by_domain(stitch_order)
            self.get_domain_Loc(self.fov_height, self.fov_width, stitch_list)

            # train the linear model
            self.lr.fit(self.scope_global_loc[np.where(self.stitch_masked == 1)],
                        self.global_loc[np.where(self.stitch_masked == 1)])

        # predict the fov coordinate of non-connect domain with trained linear area
        # for row,col in np.vstack(np.where(self.stitch_masked!=1)).T:
        #     self.global_loc[row,col] = self.lr.predict([self.scope_global_loc[row,col]])
        #     self.stitch_masked[row,col] = 1
        self.fix_unstitch_loc()
        self.cumulate_offset = np.max(self.offset_diff)
        # self.save_single_heatmap(self.offset_diff, name="offset_diff")

        self.global_loc[:, :, 0] -= np.min(self.global_loc[:, :, 0])
        self.global_loc[:, :, 1] -= np.min(self.global_loc[:, :, 1])

    # def save_single_heatmap(self, img: np.ndarray, name):
    #     """
    #     save one channel image
    #     """
    #     import matplotlib.pyplot as plt
    #     import seaborn as sns
    #
    #     sns.set(font_scale=2)
    #     fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(32, 32))
    #     cmap = sns.light_palette('purple', as_cmap=True)
    #
    #     ax1.set_title("{}".format(name))
    #     sns.heatmap(img, ax=ax1, cmap=cmap, annot=True)
    #     if self.output is not None:
    #         if os.path.exists(self.output) is not True:
    #             os.makedirs(self.output)
    #         save_path = os.path.join(self.output, "Evaluate_{}.jpg".format(name))
    #         plt.savefig(save_path)

    def _fixJitter(self, ):
        '''
        complete the horizontal and vertical offset matrix
        '''
        h_x = np.mean([self.horizontal_jitter[i, j, 0] for i in range(self.rows)
                       for j in range(self.cols) if self.horizontal_jitter[i, j, 0] != 999])
        h_y = np.mean([self.horizontal_jitter[i, j, 1] for i in range(self.rows)
                       for j in range(self.cols) if self.horizontal_jitter[i, j, 1] != 999])

        v_x = np.mean([self.vertical_jitter[i, j, 0] for i in range(self.rows)
                       for j in range(self.cols) if self.vertical_jitter[i, j, 0] != 999])
        v_y = np.mean([self.vertical_jitter[i, j, 1] for i in range(self.rows)
                       for j in range(self.cols) if self.vertical_jitter[i, j, 1] != 999])

        for i in range(self.rows):
            for j in range(self.cols):
                if self.horizontal_jitter[i, j, 0] == 999:
                    if j == 0:
                        self.horizontal_jitter[i, j] = [0, 0]
                    else:
                        if self.jitter_mask[i, j] == 999:
                            self.horizontal_jitter[i, j] = [round(h_x), round(h_y)]

                if self.vertical_jitter[i, j, 0] == 999:
                    if i == 0:
                        self.vertical_jitter[i, j] = [0, 0]
                    else:
                        if self.jitter_mask[i, j] == 999:
                            self.vertical_jitter[i, j] = [round(v_x), round(v_y)]

    def get_nearest_points(self, input_point, template_points):
        # get the 2 nearest points from the input point
        tree = spt.cKDTree(data=template_points)
        distance, index = tree.query(input_point, k=2)
        neighbor_point = template_points[index[0]]
        distance = np.abs(input_point - neighbor_point)
        return distance, neighbor_point

    def get_nearest_domain_fov(self, current_domain, stitched_loc):
        """ get the row and column index of the FOV closest to the connected domain in current connect domain """
        min_distance = 999
        nearst_domain = None
        for fov_loc in current_domain:
            distance, _ = self.get_nearest_points(fov_loc, stitched_loc)
            distance = np.linalg.norm(distance)
            if distance < min_distance:
                min_distance = distance
                nearst_domain = fov_loc
        return nearst_domain, min_distance

    def get_domain_Loc(self, height, width, stitch_list):
        ''' get the stitch location of the connect domain '''
        # self._fixJitter()
        stitch_mask = np.zeros(shape=(self.rows, self.cols))

        nearest_domain_fov = None
        for item in stitch_list:
            row, col = item
            self.stitch_masked[row, col] = 1
            if np.max(stitch_mask) == 0:
                # get the closest FOV to the stitched connected domain in current domain
                # get the closest FOV to the stitched connected domain in current domain Find the connected domain that has been spliced with the recent fov
                stitched_loc = np.where(self.stitch_masked == 1)
                stitched_loc = np.vstack(stitched_loc).T
                nearest_domain_fov, _ = self.get_nearest_domain_fov(stitch_list, stitched_loc)
                stitch_mask[row, col] = 1
                self.global_loc[row, col] = self.lr.predict([self.scope_global_loc[row, col]])[0]
            else:
                mask_list = self.neighbor(row, col, mask=stitch_mask)

                loc_list = list()
                for mask in mask_list:
                    row_src, col_src = mask
                    temp_loc = None
                    if row_src > row:
                        if self.vertical_jitter[row_src, col_src, 0] != 999:
                            temp_loc = self.global_loc[row_src, col_src] - \
                                       ([0, height] + self.vertical_jitter[row_src, col_src])
                    elif row_src < row:
                        if self.vertical_jitter[row, col, 0] != 999:
                            temp_loc = self.global_loc[row_src, col_src] + \
                                       ([0, height] + self.vertical_jitter[row, col])
                    elif col_src > col:
                        if self.horizontal_jitter[row_src, col_src, 0] != 999:
                            temp_loc = self.global_loc[row_src, col_src] - \
                                       ([width, 0] + self.horizontal_jitter[row_src, col_src])
                    elif col_src < col:
                        if self.horizontal_jitter[row, col, 0] != 999:
                            temp_loc = self.global_loc[row_src, col_src] + \
                                       ([width, 0] + self.horizontal_jitter[row, col])
                    if temp_loc is not None:
                        loc_list.append(temp_loc)

                if len(loc_list) >= 1:
                    loc_list = np.array(loc_list)
                    self.global_loc[row, col] = np.mean(loc_list, axis=0) + 0.5  # 四舍五入
                    stitch_mask[row, col] = 1
                    if len(loc_list) >= 2:
                        ptp = np.ptp(loc_list, axis=0) / 2
                        self.offset_diff[row, col] = np.around(np.linalg.norm(ptp))
                else:
                    stitch_list.append(item)
                    # glog.info("{}:{} no stitch neighbor".format(row, col))
        # adjust the location of FOV of nearest-neighbor domain to predicted location 将最近邻连通域的FOV的位置调整到预测位置
        if nearest_domain_fov is not None:
            nearest_fov_scope_loc = self.scope_global_loc[nearest_domain_fov[0], nearest_domain_fov[1]]
            offset = self.global_loc[nearest_domain_fov[0], nearest_domain_fov[1]] - \
                     self.lr.predict([nearest_fov_scope_loc])[0]
            offset = np.around(offset)
            self.global_loc[np.where(stitch_mask == 1)] -= offset.astype(np.int32)

    def getLastGlobalLoc(self, height, width):
        '''get the final stitch coordinate '''
        self._fixJitter()
        for item in self.stitch_list:
            row, col = item
            if np.max(self.stitch_mask) == 0:
                self.stitch_mask[row, col] = 1
                self.global_loc[row, col] = [0, 0]
            else:
                mask_list = self.neighbor(row, col, mask=self.stitch_mask)
                self.stitch_mask[row, col] = 1
                loc_list = list()
                for mask in mask_list:
                    row_src, col_src = mask
                    temp_loc = None
                    if row_src > row:
                        if self.vertical_jitter[row_src, col_src, 0] != 999:
                            temp_loc = self.global_loc[row_src, col_src] - \
                                       ([0, height] + self.vertical_jitter[row_src, col_src])
                    elif row_src < row:
                        if self.vertical_jitter[row, col, 0] != 999:
                            temp_loc = self.global_loc[row_src, col_src] + \
                                       ([0, height] + self.vertical_jitter[row, col])
                    elif col_src > col:
                        if self.horizontal_jitter[row_src, col_src, 0] != 999:
                            temp_loc = self.global_loc[row_src, col_src] - \
                                       ([width, 0] + self.horizontal_jitter[row_src, col_src])
                    elif col_src < col:
                        if self.horizontal_jitter[row, col, 0] != 999:
                            temp_loc = self.global_loc[row_src, col_src] + \
                                       ([width, 0] + self.horizontal_jitter[row, col])
                    if temp_loc is not None:
                        loc_list.append(temp_loc)

                if len(loc_list) >= 1:
                    loc_list = np.array(loc_list)
                    self.global_loc[row, col] = np.mean(loc_list, axis=0)
                # else:
                #     glog.info("{}:{} no stitch neighbor".format(row, col))

        self.global_loc[:, :, 0] -= np.min(self.global_loc[:, :, 0])
        self.global_loc[:, :, 1] -= np.min(self.global_loc[:, :, 1])

    def check_feature_matrix(self, feature_mask):
        """ check feature mask to get neighbor
            :return:
        """
        rows, cols = feature_mask.shape
        fix_list = []
        for row in range(rows):
            for col in range(cols):
                if self.stitch_masked[row, col] == 0:
                    it = self.neighbor_fix(row, col)
                    if it is not None:
                        fix_list.append(([row, col], it))
        return fix_list

    def check_up_stitch(self, row, col):
        """ Check whether the specified FOV can be stitched based on the offset obtained by the algorithm
        :param row:
        :param col:
        :return:
        """

        flag = False
        if self.stitch_masked[row, col] != 1 and row >= 0 and col >= 0:
            loc = []
            if col > 0 and self.horizontal_jitter[row, col, 0] != 999 and self.stitch_masked[
                row, col - 1] == 1:  # left
                x0, y0 = self.global_loc[row, col - 1] + [self.fov_width, 0] + \
                         self.horizontal_jitter[row, col]
                loc.append([x0, y0])
            if row < self.rows - 1 and self.vertical_jitter[row + 1, col, 0] != 999 and self.stitch_masked[
                row + 1, col] == 1:  # down
                x1, y1 = self.global_loc[row + 1, col] - (
                            [0, self.fov_height] + self.vertical_jitter[row + 1, col])
                loc.append([x1, y1])
            if col < self.cols - 1 and self.horizontal_jitter[row, col + 1, 0] != 999 and self.stitch_masked[
                row, col + 1] == 1:  # right
                x2, y2 = self.global_loc[row, col + 1] - (
                            [self.fov_width, 0] + self.horizontal_jitter[row, col + 1])
                loc.append([x2, y2])
            if row > 0 and self.vertical_jitter[row, col, 0] != 999 and self.stitch_masked[
                row - 1, col] == 1:  # up
                x3, y3 = self.global_loc[row - 1, col] + (
                        [0, self.fov_height] + self.vertical_jitter[row, col])
                loc.append([x3, y3])
            if len(loc) != 0:
                self.global_loc[row, col] = np.mean(loc, axis=0)
                self.stitch_masked[row, col] = 1
                flag = True  # successful stitching
        return flag

    def fix_unstitch_loc(self):
        """ The unstitching FOV is estimated to be stitching
            :return:
        """
        fix_list = self.check_feature_matrix(self.stitch_masked)
        h_mean_all = np.mean(
            [self.horizontal_jitter[i, j, :] for i in range(self.rows) for j in range(self.cols)
             if self.horizontal_jitter[i, j, 0] != 999], axis=0)
        v_mean_all = np.mean([self.vertical_jitter[i, j, :] for i in range(self.rows) for j in range(self.cols)
                              if self.vertical_jitter[i, j, 0] != 999], axis=0)
        while len(fix_list) > 0:
            for dst, src in fix_list:
                flag = self.check_up_stitch(dst[0], dst[1])
                if flag is False:
                    # Get all the offsets in this column
                    h_mean_list = [self.horizontal_jitter[row, dst[1], :] for row in range(self.rows)
                                   if self.horizontal_jitter[row, dst[1], 0] != 999]
                    h_mean = np.mean(h_mean_list, axis=0) if len(h_mean_list) > 0 else h_mean_all
                    # Get all the offsets in this row
                    v_mean_list = [self.vertical_jitter[dst[0], col, :] for col in range(self.cols)
                                   if self.vertical_jitter[dst[0], col, 0] != 999]
                    v_mean = np.mean(v_mean_list, axis=0) if len(v_mean_list) > 0 else v_mean_all

                    # Find the neighbors around the current dst
                    neighbor = self.neighbor(dst[0], dst[1], self.stitch_masked)
                    tem_loc = []
                    for src in neighbor:
                        if dst[0] < src[0]:  # down
                            x0, y0 = self.global_loc[src[0], src[1]] - [0, self.fov_height] - v_mean
                            tem_loc.append([x0, y0])
                        elif dst[0] > src[0]:  # up
                            x0, y0 = self.global_loc[src[0], src[1]] + [0, self.fov_height] + v_mean
                            tem_loc.append([x0, y0])
                        elif dst[1] < src[1]:  # right
                            x0, y0 = self.global_loc[src[0], src[1]] - [self.fov_width, 0] - h_mean
                            tem_loc.append([x0, y0])
                        elif dst[1] > src[1]:  # left
                            x0, y0 = self.global_loc[src[0], src[1]] + [self.fov_width, 0] + h_mean
                            tem_loc.append([x0, y0])
                        else:
                            glog.info('[{}-{}] fix_loc error!'.format(dst[0], dst[1]))
                    if len(tem_loc) > 0:
                        self.global_loc[dst[0], dst[1]] = np.mean(tem_loc, axis=0)
                        self.stitch_masked[dst[0], dst[1]] = 1
                        if len(tem_loc) >= 2:
                            ptp = np.ptp(tem_loc, axis=0) / 2
                            self.offset_diff[dst[0], dst[1]] = np.around(np.linalg.norm(ptp))

            fix_list = self.check_feature_matrix(self.stitch_masked)
        self.global_loc = np.around(self.global_loc, decimals=0).astype(np.int32)

        self.global_loc[:, :, 0] -= np.min(self.global_loc[:, :, 0])
        self.global_loc[:, :, 1] -= np.min(self.global_loc[:, :, 1])


if __name__ == "__main__":
    x_jitter = np.load(r"D:\02.data\temp\temp\x_c.npy")
    y_jitter = np.load(r"D:\02.data\temp\temp\y.npy")

    location_model = GlobalLocation()
    location_model.set_size(25, 28)
    location_model.set_image_shape(6105, 4608)
    location_model.set_overlap(0.1, 0)
    location_model.set_jitter(x_jitter, y_jitter)
    # 'cd' 'LS-H' 'LS-V'
    location_model.create_location('LS-V')