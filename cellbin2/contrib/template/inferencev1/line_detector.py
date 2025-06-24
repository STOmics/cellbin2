import math
import random
import copy
import numpy as np
from scipy.signal import argrelextrema
from sklearn.linear_model import LinearRegression

model = LinearRegression()


def random_color():
    b = random.randint(0, 256)
    g = random.randint(0, 256)
    r = random.randint(0, 256)
    return b, g, r


infinity = 0.00000001


def rotate(pt, angle, ori_w, ori_h, new_w, new_h):
    px, py = pt
    cx = int(new_w / 2)
    cy = int(new_h / 2)
    theta = angle
    rad = math.radians(theta)
    new_px = cx + float(px - cx) * math.cos(rad) + float(py - cy) * math.sin(rad)
    new_py = cy + -(float(px - cx) * math.sin(rad)) + float(py - cy) * math.cos(rad)
    x_offset, y_offset = (ori_w - new_w) / 2, (ori_h - new_h) / 2
    new_px += x_offset
    new_py += y_offset
    return int(new_px), int(new_py)


class Line(object):
    def __init__(self, ):
        self.coefficient = None
        self.bias = None
        self.index = 0

    def two_points(self, shape):
        h, w = shape
        if self.coefficient >= 0:
            pt0 = self.get_point_by_x(0)
            pt1 = self.get_point_by_x(w)
        else:
            pt0 = self.get_point_by_y(0)
            pt1 = self.get_point_by_y(h)
        return [pt0, pt1]

    def set_coefficient_by_rotation(self, rotation):
        self.coefficient = math.tan(math.radians(rotation))

    def init_by_point_pair(self, pt0, pt1):
        x0, y0 = pt0
        x1, y1 = pt1
        if x1 > x0:
            self.coefficient = (y1 - y0) / (x1 - x0)
        elif x1 == x0:
            self.coefficient = (y0 - y1) / infinity
        else:
            self.coefficient = (y0 - y1) / (x0 - x1)
        self.bias = y0 - self.coefficient * x0

    def init_by_point_k(self, pt0, k):
        x0, y0 = pt0
        self.coefficient = k
        self.bias = y0 - k * x0

    def rotation(self, ):
        return math.degrees(math.atan(self.coefficient))

    def get_point_by_x(self, x):
        return [x, self.coefficient * x + self.bias]

    def get_point_by_y(self, y):
        return [(y - self.bias) / self.coefficient, y]

    def line_rotate(self, angle, ori_w, ori_h, new_w, new_h):
        shape = (new_h, new_w)
        p0, p1 = self.two_points(
            shape=shape
        )
        p0_new = rotate(
            p0,
            angle,
            ori_w, ori_h, new_w, new_h
        )
        p1_new = rotate(
            p1,
            angle,
            ori_w, ori_h, new_w, new_h
        )
        self.init_by_point_pair(p0_new, p1_new)
        return self


class TrackLineDetector(object):
    def __init__(self):
        self.grid = 100

    def generate(self, arr):
        """
        This algorithm will not work the angle of image is more than 8 degree

        Args:
            arr (): 2D array in uint 8 or uint 16

        Returns:
            h_lines: horizontal line
            v_lines: vertical line

        """
        h_lines, v_lines = [], []

        # horizontal direction
        horizontal_candidate_pts = self.create_candidate_pts(arr, 'x')
        h_angle = self.integer_angle(horizontal_candidate_pts, 'x')
        if h_angle != -1000:
            horizontal_pts = self.select_pts_by_integer_angle(horizontal_candidate_pts, h_angle, tolerance=1)
            if len(horizontal_pts) != 0:
                horizontal_color_pts = self.classify_points(horizontal_pts, h_angle, tolerance=1)
                h_lines = self.points_to_line(horizontal_color_pts, tolerance=3)

        # vertical direction
        vertical_candidate_pts = self.create_candidate_pts(arr, 'y')
        v_angle = self.integer_angle(vertical_candidate_pts, 'y')
        if v_angle != -1000:
            vertical_pts = self.select_pts_by_integer_angle(vertical_candidate_pts, v_angle, tolerance=1)
            if len(vertical_pts) != 0:
                vertical_color_pts = self.classify_points(vertical_pts, v_angle, tolerance=1)
                v_lines = self.points_to_line(vertical_color_pts, tolerance=3)

        return h_lines, v_lines

    @staticmethod
    def points_to_line(dct, tolerance=2):
        lines = list()
        for k, v in dct.items():
            # Less than two fits do not form a straight line
            if len(v) > tolerance:
                tmp = np.array(v)
                model.fit(tmp[:, 0].reshape(-1, 1), tmp[:, 1])
                line = Line()
                # A point, combined with coef to fit a straight line
                line.init_by_point_k(v[0], model.coef_[0])
                lines.append(line)
        return lines

    def classify_points(self, candidate_pts, base_angle, tolerance=2):
        pts = copy.copy(candidate_pts)
        ind = 0
        dct = dict()
        while (len(pts) > 1):
            pts_, index = self.angle_line(base_angle, pts, tolerance)
            # Put the points of the same straight line found into DCT
            dct[ind] = pts_
            # Delete the found points based on the index returned in the anglenline
            pts = np.delete(np.array(pts), index, axis=0).tolist()
            ind += 1
        # Return the DCT of the classification of stored points
        return dct

    @staticmethod
    def angle_line(angle, points, tolerance=2):
        # Find a point on a straight line with points [0]
        count = len(points)
        orignal_point = points[0]
        points_ = [points[0]]
        index = [0]
        for i in range(1, count):
            p = points[i]
            line = Line()
            line.init_by_point_pair(orignal_point, p)
            diff = abs(line.rotation() - angle)
            diff = (diff > 90) and (180 - diff) or diff
            if diff < tolerance:
                points_.append(p)
                index.append(i)
        # Return all points of this line, along with their corresponding index numbers.
        # The index is used to delete the points found in the array later
        return points_, index

    @staticmethod
    def select_pts_by_integer_angle(candidate_pts, base_angle, tolerance=2):
        x_count = len(candidate_pts)
        # Pts is used to store all points in that direction
        pts = list()
        for i in range(0, x_count - 1):
            if len(candidate_pts[i]) > 100:
                continue
            # Traverse all sampling areas
            pts_start = candidate_pts[i]
            pts_end = candidate_pts[i + 1]
            # Traverse all points in pts_start
            for p0 in pts_start:
                # Traverse all points in pts_end and calculate the distance between p0 and each point
                d = [math.sqrt(pow(p0[0] - p1[0], 2) + pow(p0[1] - p1[1], 2)) for p1 in pts_end]
                # Take absolute value
                d_ = np.abs(d)
                # Find the index of the shortest distance
                ind = np.where(d_ == np.min(d_))[0]
                line = Line()
                # Calculate angle
                line.init_by_point_pair(p0, pts_end[ind[0]])
                # If the angle is less than tol
                if abs(line.rotation() - base_angle) <= tolerance: pts.append(p0)
        return pts

    @staticmethod
    def integer_angle(pts, derection='x'):
        angle = -1000
        x_count = len(pts)
        # Angles will save the angles of all sampling areas in that direction
        angles = list()
        for i in range(0, x_count - 1):
            if len(pts[i]) > 100:
                continue
            # Two adjacent sampling regions
            pts_start = pts[i]
            pts_end = pts[i + 1]
            for p0 in pts_start:
                # A point p0 in pts_start corresponds to the Euclidean distance of all points in pts_end
                d = [math.sqrt(pow(p0[0] - p1[0], 2) + pow(p0[1] - p1[1], 2)) for p1 in pts_end]
                # Take absolute value
                d_ = np.abs(d)
                # Find the index of the minimum distance to the point of pts_start
                ind = np.where(d_ == np.min(d_))[0]
                line = Line()

                line.init_by_point_pair(p0, pts_end[ind[0]])
                # The rotation method uses the coef obtained above to calculate the angle of the line
                # And record the angle
                angles.append(round(line.rotation()))
        if len(angles) != 0:
            x = np.array(angles) - np.min(angles)
            # Obtain the angle in that direction (the angle with the most occurrences)
            angle = np.argmax(np.bincount(x)) + np.min(angles)
        return angle

    def create_candidate_pts(self, mat, derection='x'):
        pts = list()
        h, w = mat.shape
        # direction x -> h
        # direction y -> w
        counter = (derection == 'x' and h or w)
        # self.grid -> defined by user
        for i in range(0, counter, self.grid):
            # Set t to x or y of the current sampling interval
            t = i + self.grid / 2
            if derection == 'x':
                # Region -> i to i+sampling interval region
                region_mat = mat[i: i + self.grid, :w]
                # If the area is not of the specified sampling length, continue
                if region_mat.shape[0] != self.grid:
                    continue
                # Summing up the pixels in the y direction and
                # dividing it by the specified sampling interval can be seen as normalization
                line = np.sum(region_mat, axis=0) / self.grid
            else:
                # The handling here is basically the same as above, except that this is the case for direction y
                region_mat = mat[:h, i: i + self.grid]
                if region_mat.shape[1] != self.grid:
                    continue
                line = np.sum(region_mat, axis=1) / self.grid
            # Find the extreme value (minimum value) on this line
            p = argrelextrema(line, np.less_equal, order=100)
            # print(p[0].shape)
            if derection == 'x':
                pt = [[p, t] for p in p[0]]
            else:
                pt = [[t, p] for p in p[0]]
            # All points in that direction are saved in pts
            pts.append(pt)
        return pts


def main():
    import cv2
    image_path = r"D:\Data\tmp\Y00035MD\Y00035MD\Y00035MD_0000_0004_2023-01-30_15-50-41-868.tif"
    arr = cv2.imread(image_path, -1)
    ftl = TrackLineDetector()
    result = ftl.generate(arr)


if __name__ == '__main__':
    main()
