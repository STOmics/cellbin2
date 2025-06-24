"""
细胞轮廓点写入cgef文件
版本：V1.0.1
"""

import gzip
import json
import logging
import math
import os
import random
import struct
import sys
import time
from itertools import chain
from optparse import OptionParser

import h5py
import mapbox_earcut as earcut
import numpy as np

# 50w 个细胞
# file = 'D:/gef/FP200000616TL_A3C4.cellbin.gef'
# output_folder = 'D:/ng_local_data/cell_border/2D_data/FP200000616TL_A3C4'
# 5w 个细胞
# file = 'D:/gef/SS200000135TL_D1.cellbin.gef'
# output_folder = 'D:/ng_local_data/cell_border/2D_data/SS200000135TL_D1'
# 100w 个细胞
# file = 'D:/gef/SS200000144TR_C1E4.cellbin.gef'
# output_folder = 'D:/ng_local_data/cell_border/2D_data/SS200000144TR_C1E4'
# 100w 个细胞
# file = 'D:/gef/SS200000144TR_C1E4.cellbin-new.gef'
# output_folder = 'D:/ng_local_data/cell_border/2D_data/SS200000144TR_C1E4'
# TODO 需要另外输入info中设置properties，properties中内容需要可配置
# 细胞轮廓点数量，固定值
final_border_length = 16

# TODO 需要改成读取gef文件中的属性名称对应的数据下标
attr_x_index = 1
attr_y_index = 2
attr_gene_count_index = 4
attr_exp_count_index = 5
attr_dnb_count_index = 6
attr_area_index = 7
attr_cell_type_id_index = 8
attr_cluster_id_index = 9

# h5文件指针
H5_FILE_POINT: h5py.File


def create_group(group_name):
    """
    h5文件创建group

    :param group_name: group 名称
    :return: group 对象
    """
    global H5_FILE_POINT
    if H5_FILE_POINT.get(group_name):
        del H5_FILE_POINT[group_name]
    return H5_FILE_POINT.create_group(group_name)


def set_attrs(group: h5py.Group, key, values):
    """
    h5 group 设置属性

    :param group: group 名称
    :param key: 属性名
    :param values: 属性值
    :return:
    """
    group.attrs[key] = values


def create_dataset(group: h5py.Group, name, data, dtype=None):
    """
    h5 group 创建数据集
    :param group: group名称
    :param name: 数据集名称
    :param data: 数据
    :param dtype: 数据类型 **暂未用到
    :return:
    """
    group.create_dataset(name, data=data)


def print_running_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        res = func(*args, **kwargs)
        end_time = time.time()
        run_time = end_time - start_time
        print('Running time of "{0}": {1}'.format(func.__name__, run_time))
        return res

    return wrapper


# 创建info json文件
@print_running_time
def create_info_json_2d(resolution, lower_bound, upper_bound, chunk_size, grid_shape, empty_chunk, output_folder,
                        group):
    info = {'@type': 'neuroglancer_annotations_v1',
            'annotation_type': 'CELL_SHAPE',
            'dimensions': {'x': [resolution, 'nm'],
                           'y': [resolution, 'nm']},
            'by_id': {'key': 'by_id'},
            'relationships': [],
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'properties': [
                {'id': 'id', 'type': 'uint32'},
                {'id': 'geneCount', 'type': 'uint32'},
                {'id': 'expCount', 'type': 'uint32'},
                {'id': 'dnbCount', 'type': 'uint32'},
                {'id': 'area', 'type': 'uint32'},
                {'id': 'cellTypeID', 'type': 'uint32'},
                {'id': 'clusterID', 'type': 'uint32'}
            ],
            'spatial': [
                {'chunk_size': chunk_size,
                 'grid_shape': grid_shape,
                 'key': 'L0',
                 'limit': 10000
                 }
            ],
            'emptyChunk': empty_chunk
            }
    # if not os.path.exists(output_folder):  # 判断当前路径是否存在，没有则创建文件夹
    #     os.makedirs(output_folder)
    # with open(output_folder + '/info', 'w') as write_f:
    #     write_f.write(json.dumps(info, indent=4, ensure_ascii=False))

    set_attrs(group, 'info', json.dumps(info))


def divChunk(cell_count, bounding_size_x, bounding_size_y):
    grid_shape = [8, 8]
    min_dim = 0

    min_size = bounding_size_x
    max_size = bounding_size_y

    if bounding_size_x > bounding_size_y:
        min_size = bounding_size_y
        max_size = bounding_size_x
        min_dim = 1
    # 1.415为根号2的近似值
    target_size = 1.415 * min_size
    pre_div_times = 0

    while max_size > target_size:
        max_size /= 2
        pre_div_times += 1
    pre_div_chunk_count = pow(2, pre_div_times)

    div_depth = 0
    chunk_count = pre_div_chunk_count * pow(4, div_depth)
    chunk_cell_count = int(cell_count / chunk_count)
    while chunk_cell_count > 5000:
        div_depth += 1
        chunk_count = pre_div_chunk_count * pow(4, div_depth)
        chunk_cell_count = int(cell_count / chunk_count)

    if min_dim == 0:
        grid_shape[0] = pow(2, div_depth)
        grid_shape[1] = pre_div_chunk_count * pow(2, div_depth)
    else:
        grid_shape[1] = pow(2, div_depth)
        grid_shape[0] = pre_div_chunk_count * pow(2, div_depth)
    return grid_shape


@print_running_time
def getCellInfoList(file, output_folder):
    with h5py.File(file, mode='r') as gef_file:

        resolution = int(gef_file.attrs[['resolution'][0]])
        cell_data = gef_file.get('cellBin/cell')
        lower_bound = [float(cell_data.attrs['minX'][0]), float(cell_data.attrs['minY'][0])]
        upper_bound = [float(cell_data.attrs['maxX'][0]), float(cell_data.attrs['maxY'][0])]

        cell_count = len(cell_data)

        bounding_size_x = upper_bound[0] - lower_bound[0]
        bounding_size_y = upper_bound[1] - lower_bound[1]

        grid_shape = divChunk(cell_count, bounding_size_x, bounding_size_y)
        # 根据分块网格比例以及边界值，计算分块大小
        chunk_size = [bounding_size_x / grid_shape[0], bounding_size_y / grid_shape[1]]
        cell_basic_info_list = list(map(list, cell_data[:]))

        if len(cell_basic_info_list[0]) < 10:
            global attr_x_index
            global attr_y_index
            global attr_area_index
            global attr_gene_count_index
            global attr_dnb_count_index
            global attr_cell_type_id_index
            global attr_cluster_id_index
            attr_x_index -= 1
            attr_y_index -= 1
            attr_gene_count_index -= 1
            attr_dnb_count_index -= 1
            attr_area_index -= 1
            attr_cell_type_id_index -= 1
            attr_cluster_id_index -= 1

        border_data = gef_file.get('cellBin/cellBorder')
        cell_border_list = []

        # 用于测试导出指定个数的细胞
        targe_cell_count = cell_count
        # targe_cell_count = 10

        for i in range(targe_cell_count):
            cell_border_list.append([list(j) for j in border_data[i]])

        # 获取所有的细胞列表
        all_cell_list = []
        if len(cell_border_list[0]) > 31:
            for i in range(targe_cell_count):
                two_cell_border_info = split_32points_cell_border(cell_border_list[i])
                is_split_cell = two_cell_border_info[1] is not None
                cell1 = create_cell_dict(cell_basic_info_list[i][0], cell_basic_info_list[i],
                                         CellBorder(i, two_cell_border_info[0], is_split_cell))
                # 随机插入细胞，前台渲染部分渲染时才能避免出现若隐若现的分块形状
                all_cell_list.insert(int(random.random() * len(all_cell_list)), cell1)

                if is_split_cell:
                    cell2 = create_cell_dict(cell_basic_info_list[i][0], cell_basic_info_list[i],
                                             CellBorder(i, two_cell_border_info[1], True))
                    all_cell_list.insert(int(random.random() * len(all_cell_list)), cell2)
        else:
            for i in range(targe_cell_count):
                border_data = cell_border_list[i]
                valid_len = get_valid_border_len(border_data)
                cell = create_cell_dict(i, cell_basic_info_list[i], CellBorder(i, border_data[:valid_len]))
                all_cell_list.insert(int(random.random() * len(all_cell_list)), cell)

        # 将所有细胞根据位置放入分块中
        chunk_cells_dict = create_chunk_grids_dict(grid_shape, output_folder)

        for cell in all_cell_list:
            append_cell_to_chunk(cell, chunk_cells_dict, lower_bound, chunk_size, grid_shape)

        group = create_group('codedCellBlock/L0')

        empty_chunk = {"0": []}
        empty_chunk_list = empty_chunk["0"]
        for chunk_key in chunk_cells_dict:
            chunk = chunk_cells_dict[chunk_key]
            if len(chunk.cell_list) > 0:
                # chunk_cells_dict[chunk_key].output_byte_file()
                chunk_cells_dict[chunk_key].write_h5(group)
            else:
                empty_chunk_list.append(chunk_key.replace("_", ","))

        create_info_json_2d(resolution, lower_bound, upper_bound, chunk_size, grid_shape, empty_chunk, output_folder,
                            H5_FILE_POINT['codedCellBlock'])


# 将32个顶点的细胞轮廓点，拆分成两个16个顶点的细胞轮廓点
# 需要考虑葫芦形，S形等怪异形状细胞的拆分。
# @print_running_time
def split_32points_cell_border(border_data):
    border_data_len = len(border_data)
    # 处理细胞轮廓点的数量大于32个的情况，多于32个点直接舍弃
    if border_data_len > 31:
        border_data = border_data[:32]
    valid_len = get_valid_border_len(border_data)
    valid_border_point_list = border_data[:valid_len]
    # 如果点数超过30，就做剔除，留下30个轮廓点。因为30个点的细胞轮廓拆分出来后，有两个点是复制出来的，刚好能拆成两个16个点的细胞
    simply_point_list = valingam_whyatt_shape_simplification(30, valid_border_point_list, False)["point_list"]
    point_length = len(simply_point_list)
    if point_length > 16:

        vertex = np.array(simply_point_list)
        rings = np.array([point_length])

        # 进行耳切，找到切分点。切出来的两个细胞轮廓点的数量不能大于16
        index_arr = earcut.triangulate_float32(vertex, rings)
        triangle_count = int(len(index_arr) / 3)

        slit_point_1_index = None
        slit_point_2_index = None
        is_got_split_line = False
        # 找到earcut后的中间切线。为了将葫芦形或者S形的细胞拆分成正确的两个细胞，切线不能超过原有的细胞轮廓。
        # 耳切法三角剖分后，找到任意一条线，能将细胞切成两个轮廓点小于等于16的小细胞。
        # 耳切后，得到数量为triangle_count的三角面，index_arr中3个为一组代表一个三角面的下标顶点
        for i in range(triangle_count):

            slit_point_1_index = int(index_arr[i * 3])
            slit_point_2_index = int(index_arr[i * 3 + 1])
            strid = abs(slit_point_1_index - slit_point_2_index)
            if (strid < 16) and (point_length - strid < 16):
                is_got_split_line = True
                break

            slit_point_1_index = int(index_arr[i * 3])
            slit_point_2_index = int(index_arr[i * 3 + 2])
            strid = abs(slit_point_1_index - slit_point_2_index)
            if (strid < 16) and (point_length - strid < 16):
                is_got_split_line = True
                break

            slit_point_1_index = int(index_arr[i * 3 + 2])
            slit_point_2_index = int(index_arr[i * 3 + 1])
            strid = abs(slit_point_1_index - slit_point_2_index)
            if (strid < 16) and (point_length - strid < 16):
                is_got_split_line = True
                break

        if is_got_split_line:
            start_index = min(slit_point_1_index, slit_point_2_index)
            end_index = max(slit_point_1_index, slit_point_2_index)

            start_list = simply_point_list[start_index:end_index + 1]
            end_list = simply_point_list[end_index:] + simply_point_list[:start_index + 1]
            # 截取到顶点数量比较多的细胞轮廓，放到第一个。渲染时，LOD等级切换到6个顶点时，可不渲染小的那半部分的细胞，提高渲染性能
            if len(end_list) > len(start_list):
                start_list, end_list = end_list, start_list
        # 如果耳切后，没有适合的边，意味着这个细胞形状为凸多边形。
        else:
            # 将前16个轮廓点抽出作为第一个切分细胞
            start_list = border_data[:16]
            # 第16个点到最后一个有效点为第二个细胞轮廓，另外还要将第一个点复制到第二个切分的细胞。
            end_list = border_data[15:len(simply_point_list)]
            end_list.append(border_data[0])
    else:
        start_list = border_data[:valid_len]
        end_list = None

    return [start_list, end_list]


# 获取有效的细胞轮廓点数量
# @print_running_time
def get_valid_border_len(border_data):
    length = len(border_data)
    border_vertex_len = length
    # 计算细胞轮廓有效点的数量
    for index in range(length):
        pos_arr = border_data[index]
        if index < length - 1:
            next_pos_arr = border_data[index + 1]
            if ((pos_arr[0] == 0) and (pos_arr[1] == 0) and (next_pos_arr[0] == 0) and (next_pos_arr[1] == 0)) \
                    or ((pos_arr[0] == 32767) and (pos_arr[1] == 32767) and
                        (next_pos_arr[0] == 32767) and (next_pos_arr[1] == 32767)):
                border_vertex_len = index
                break
        else:
            if ((pos_arr[0] == 32767) and (pos_arr[1] == 32767)) \
                    or ((pos_arr[0] == 0) and (pos_arr[1] == 0)):
                border_vertex_len = index
    return border_vertex_len


def append_cell_to_chunk(cell, chunk_cells_dict, lower_bound, chunk_size, grid_shape):
    chunk_x_index = min(int((cell['x'] - lower_bound[0]) / chunk_size[0]), grid_shape[0] - 1)
    chunk_y_index = min(int((cell['y'] - lower_bound[1]) / chunk_size[1]), grid_shape[1] - 1)
    chunk_key = str(chunk_x_index) + "_" + str(chunk_y_index)
    chunk_cells_dict[chunk_key].cell_list.append(cell)


# 创建分块字典 key:String 例如：0_0_0  value: list<Cell>
@print_running_time
def create_chunk_grids_dict(_grid_shape, output_folder):
    result_dict = dict()
    # byte_file_dir = output_folder + '/L0'
    # if not os.path.exists(byte_file_dir):  # 判断当前路径是否存在，没有则创建文件夹
    #     os.makedirs(byte_file_dir)

    byte_file_dir = 'codedCellBlock/L0'
    for x_index in range(_grid_shape[0]):
        for y_index in range(_grid_shape[1]):
            cur_key = str(x_index) + "_" + str(y_index)
            result_dict[cur_key] = Chunk(cur_key, byte_file_dir, [])

    return result_dict


# @print_running_time
def create_cell_dict(_id, basic_info, cell_border):
    return {
        'id': _id,
        'x': basic_info[attr_x_index],
        'y': basic_info[attr_y_index],
        'gene_count': basic_info[attr_gene_count_index],
        'dnb_count': basic_info[attr_dnb_count_index],
        'area': basic_info[attr_area_index],
        'cell_type_id': basic_info[attr_cell_type_id_index],
        'exp_count': basic_info[attr_exp_count_index],
        'cluster_id': basic_info[attr_cluster_id_index],
        'cell_border': cell_border
    }


# @print_running_time
def get_line_distance(point1, point2):
    return math.sqrt(pow(float(point2[0] - point1[0]), 2) + pow(float(point2[1] - point1[1]), 2))


# @print_running_time
def get_triangle_area(point1, point2, point3):
    a = get_line_distance(point1, point2)
    b = get_line_distance(point2, point3)
    c = get_line_distance(point1, point3)
    p = (a + b + c) / 2
    return math.sqrt(abs(p * (p - a) * (p - b) * (p - c)))


# @print_running_time
# 保留有效面积 (Visvalingam-Whyatt) 算法，剔除部分点。保留target_len长度的list
def valingam_whyatt_shape_simplification(target_count, point_list, is_need_earcut=True):
    point_list_len = len(point_list)
    if point_list_len <= target_count:
        return {
            "vertex_index": None,
            "point_list": point_list
        }
    else:
        point_obj_dict = {}
        for index in range(point_list_len):
            pre_node_key = (index - 1 + point_list_len) % point_list_len
            next_node_key = (index + 1 + point_list_len) % point_list_len
            pre_point = point_list[pre_node_key]
            next_point = point_list[next_node_key]
            cur_point = point_list[index]
            point_area = get_triangle_area(pre_point, cur_point, next_point)
            point_obj = {'index': index,
                         'area': point_area,
                         'pos': cur_point,
                         'pre_node_key': pre_node_key,
                         'next_node_key': next_node_key,
                         }
            point_obj_dict[index] = point_obj

        while point_list_len > target_count:
            min_area = float('inf')
            min_area_index = -1
            for key in point_obj_dict:
                if min_area_index == -1:
                    min_area_index = point_obj_dict[key]['index']
                if point_obj_dict[key]['area'] < min_area:
                    min_area = point_obj_dict[key]['area']
                    min_area_index = point_obj_dict[key]['index']

            pre_node_key = point_obj_dict[min_area_index]['pre_node_key']
            next_node_key = point_obj_dict[min_area_index]['next_node_key']

            pre_node = point_obj_dict[pre_node_key]
            next_node = point_obj_dict[next_node_key]
            pre_node['next_node_key'] = next_node_key
            next_node['pre_node_key'] = pre_node_key

            pre_point = point_list[pre_node['pre_node_key']]
            next_point = point_list[pre_node['next_node_key']]
            cur_point = point_list[pre_node_key]
            pre_node['area'] = get_triangle_area(pre_point, cur_point, next_point)

            pre_point = point_list[next_node['pre_node_key']]
            next_point = point_list[next_node['next_node_key']]
            cur_point = point_list[next_node_key]
            next_node['area'] = get_triangle_area(pre_point, cur_point, next_point)

            point_obj_dict.pop(min_area_index)

            point_list_len -= 1

        point_pos_list = []
        point_obj_list = []
        for key in point_obj_dict:
            point_pos_list.append(point_obj_dict[key]['pos'])
            point_obj_list.append(point_obj_dict[key])

        vertex = np.array(point_pos_list)
        rings = np.array([len(point_pos_list)])

        lod_result = []
        if is_need_earcut:
            index_arr_lod = earcut.triangulate_float32(vertex, rings)
            for index in index_arr_lod:
                lod_result.append(point_obj_list[index]['index'])

        return {
            "vertex_index": lod_result,
            "point_list": point_pos_list
        }


class CellBorder:
    def __init__(self, _id, border_data, is_split_cell=False):
        self.id = _id
        self.border_data = border_data
        self.is_split_cell = is_split_cell
        self.border_vertex_len = len(border_data)
        for index in range(self.border_vertex_len, final_border_length):
            self.border_data.append(self.border_data[self.border_vertex_len - 1])

    def get_border_points_bytes(self):
        points_bytes = struct.pack('<32h', *list(chain(*self.border_data)))
        return points_bytes

    # @print_running_time
    def get_LOD_vertices_index(self):
        # 将有效轮廓点的二维数组降维成一维数组，最多有16个点
        vertex = np.array(self.border_data[0:self.border_vertex_len])
        rings = np.array([self.border_vertex_len])
        index_arr_lod_level_1 = earcut.triangulate_float32(vertex, rings)

        offset = 0
        vertices_index_arr = [0] * 96
        for vertex_index in index_arr_lod_level_1:
            vertices_index_arr[offset] = vertex_index
            offset += 1

        border_point_arr = self.border_data[0:self.border_vertex_len]
        if self.border_vertex_len > 8:
            lod_8_obj = valingam_whyatt_shape_simplification(8, border_point_arr)
            index_arr_lod_level_2 = lod_8_obj["vertex_index"]
        else:
            index_arr_lod_level_2 = index_arr_lod_level_1

        if self.border_vertex_len > 4:
            index_arr_lod_level_3 = valingam_whyatt_shape_simplification(4, border_point_arr)["vertex_index"]
        else:
            index_arr_lod_level_3 = index_arr_lod_level_1

        offset = 42
        for vertex_index in index_arr_lod_level_2:
            vertices_index_arr[offset] = vertex_index
            offset += 1

        offset = 60
        for vertex_index in index_arr_lod_level_3:
            vertices_index_arr[offset] = vertex_index
            offset += 1
        # 细胞是否被拆分的标识
        if self.is_split_cell:
            vertices_index_arr[66] = 1

        vertices_buffer_arr = [0] * 16
        for index in range(len(vertices_index_arr)):
            bits_to_int_arr_index = index // 8
            move_left_num = (index % 8 * 4)
            cur_add_num = vertices_index_arr[index] << move_left_num
            vertices_buffer_arr[bits_to_int_arr_index] = vertices_buffer_arr[bits_to_int_arr_index] + cur_add_num

        points_bytes = struct.pack('<16L', *vertices_buffer_arr)
        return points_bytes


class Chunk:
    def __init__(self, _key, _folder_dir, _cell_list):
        self.key = _key
        self.folder_dir = _folder_dir
        self.cell_list = _cell_list

    def output_byte_file(self):

        file_dir = self.folder_dir + "/" + self.key
        with open(file_dir, mode='wb') as f_out:
            total_count = len(self.cell_list)
            cell_list = self.cell_list
            bytes_out = struct.pack('<Q', total_count)
            id_arr = []
            for idx in range(total_count):
                x = cell_list[idx]['x']
                y = cell_list[idx]['y']
                cell_type_id = cell_list[idx]['cell_type_id']
                exp_count = cell_list[idx]['exp_count']
                gene_count = cell_list[idx]['gene_count']
                dnb_count = cell_list[idx]['dnb_count']
                cluster_id = cell_list[idx]['cluster_id']
                area = cell_list[idx]['area']
                cell_id = cell_list[idx]['id']
                id_arr.append(cell_id)

                pt_buf = struct.pack('<2f7I', x, y, cell_id, gene_count, exp_count, dnb_count, area, cell_type_id,
                                     cluster_id)
                bytes_out += pt_buf

            id_buf = struct.pack('<' + str(total_count) + 'Q', *id_arr)
            bytes_out += id_buf

            for index in range(total_count):
                cell_border = self.cell_list[index]['cell_border']
                buf = cell_border.get_border_points_bytes()
                bytes_out += buf

            for index in range(total_count):
                cell_border = self.cell_list[index]['cell_border']
                buf = cell_border.get_LOD_vertices_index()
                bytes_out += buf

            # TODO bytes_out拼接效率很低，可能是bytes_out拼接时长度不定，需要不定时扩展长度，导致性能下降。需要做性能优化
            f_out.write(gzip.compress(bytes_out))

    def write_h5(self, group):
        total_count = len(self.cell_list)
        cell_list = self.cell_list
        bytes_out = struct.pack('<Q', total_count)
        id_arr = []
        for idx in range(total_count):
            x = cell_list[idx]['x']
            y = cell_list[idx]['y']
            cell_type_id = cell_list[idx]['cell_type_id']
            exp_count = cell_list[idx]['exp_count']
            gene_count = cell_list[idx]['gene_count']
            dnb_count = cell_list[idx]['dnb_count']
            cluster_id = cell_list[idx]['cluster_id']
            area = cell_list[idx]['area']
            cell_id = cell_list[idx]['id']
            id_arr.append(cell_id)

            pt_buf = struct.pack('<2f7I', x, y, cell_id, gene_count, exp_count, dnb_count, area, cell_type_id,
                                 cluster_id)
            bytes_out += pt_buf

        id_buf = struct.pack('<' + str(total_count) + 'Q', *id_arr)
        bytes_out += id_buf

        for index in range(total_count):
            cell_border = self.cell_list[index]['cell_border']
            buf = cell_border.get_border_points_bytes()
            bytes_out += buf

        for index in range(total_count):
            cell_border = self.cell_list[index]['cell_border']
            buf = cell_border.get_LOD_vertices_index()
            bytes_out += buf

        group.create_dataset(self.key, data=np.void(gzip.compress(bytes_out)))


def f_main(cgeffile):
    global H5_FILE_POINT
    H5_FILE_POINT = h5py.File(cgeffile, "r+")
    getCellInfoList(cgeffile, None)
    H5_FILE_POINT.close()


def main():
    try:
        Usage = """
        %prog
        -i <Cellbin gef>
        -o <output Path>

        return gene expression matrix under cells with labels
        """
        parser = OptionParser(Usage)
        parser.add_option("-n", dest="sampleid", help="SampleID for input data. ")
        parser.add_option("-i", dest="cgeffile", help="Path of cgef file. ")
        parser.add_option("-o", dest="outpath", help="Output directory. ")
        opts, args = parser.parse_args()

        if not opts.cgeffile or not os.path.exists(opts.cgeffile):
            logging.error("Inputs are not correct")
            sys.exit(not parser.print_usage())

        # outpath = os.path.join(opts.outpath, 'cellbin_blocks')
        # os.makedirs(outpath, exist_ok=True)
        # getCellInfoList(opts.cgeffile, outpath)

        f_main(opts.cgeffile)
    except Exception as e:
        logging.error(e)
        pass


if __name__ == "__main__":
    main()
