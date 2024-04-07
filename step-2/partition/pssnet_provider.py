"""
    PSSNet: Planarity-sensible Semantic Segmentation of Large-scale Urban Meshes
    https://www.sciencedirect.com/science/article/pii/S0924271622003355
    2023 Weixiao GAO, Liangliang Nana, Bas Boom, Hugo Ledoux

    functions for writing and reading features and graph
"""

import os
import sys
import random
import glob
from plyfile import PlyData, PlyElement
import numpy as np
import pandas as pd
import h5py
from sklearn.neighbors import NearestNeighbors

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(DIR_PATH, '..'))
import colorsys
from sklearn.decomposition import PCA


# ------------------------------------------------------------------------------
def partition2ply(filename, xyz, components):
    """write a ply with random colors for each components"""
    random_color = lambda: random.randint(0, 255)
    color = np.zeros(xyz.shape)
    for i_com in range(0, len(components)):
        color[components[i_com], :] = [random_color(), random_color()
            , random_color()]
    prop = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1')
        , ('green', 'u1'), ('blue', 'u1')]
    vertex_all = np.empty(len(xyz), dtype=prop)
    for i in range(0, 3):
        vertex_all[prop[i][0]] = xyz[:, i]
    for i in range(0, 3):
        vertex_all[prop[i + 3][0]] = color[:, i]
    ply = PlyData([PlyElement.describe(vertex_all, 'vertex')], text=True)
    ply.write(filename)

def prediction2ply_withlabels(filename, xyz, truth, prediction, n_label, dataset):
    """write a ply with colors for each class"""
    if len(prediction.shape) > 1 and prediction.shape[1] > 1:
        prediction = np.argmax(prediction, axis=1)
    color = np.zeros(xyz.shape)
    for i_label in range(0, n_label + 1):
        color[np.where(prediction == i_label), :] = get_color_from_label(i_label, dataset)
    prop = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('truth', 'i4'), ('pred', 'i4')]
    vertex_all = np.empty(len(xyz), dtype=prop)
    for i in range(0, 3):
        vertex_all[prop[i][0]] = xyz[:, i]
    for i in range(0, 3):
        vertex_all[prop[i + 3][0]] = color[:, i]
    vertex_all[prop[6][0]] = truth
    vertex_all[prop[7][0]] = prediction
    ply = PlyData([PlyElement.describe(vertex_all, 'vertex')], text=True)
    ply.write(filename)

def partition2ply_with_segids(filename, xyz, components, in_components):
    """write a ply with random colors for each components"""
    random_color = lambda: random.randint(0, 255)
    color = np.zeros(xyz.shape)
    for i_com in range(0, len(components)):
        color[components[i_com], :] = [random_color(), random_color(), random_color()]
    prop = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1')
    , ('green', 'u1'), ('blue', 'u1'), ('point_segment_id', 'i4')]
    vertex_all = np.empty(len(xyz), dtype=prop)
    for i in range(0, 3):
        vertex_all[prop[i][0]] = xyz[:, i]
    for i in range(0, 3):
        vertex_all[prop[i+3][0]] = color[:, i]
    vertex_all[prop[6][0]] = in_components
    ply = PlyData([PlyElement.describe(vertex_all, 'vertex')], text=True)
    ply.write(filename)
# ------------------------------------------------------------------------------
def geof2ply(filename, xyz, geof):
    """write a ply with colors corresponding to geometric features"""
    color = np.array(255 * geof[:, [0, 1, 3]], dtype='uint8')
    prop = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    vertex_all = np.empty(len(xyz), dtype=prop)
    for i in range(0, 3):
        vertex_all[prop[i][0]] = xyz[:, i]
    for i in range(0, 3):
        vertex_all[prop[i + 3][0]] = color[:, i]
    ply = PlyData([PlyElement.describe(vertex_all, 'vertex')], text=True)
    ply.write(filename)


# ------------------------------------------------------------------------------
def prediction2ply(filename, xyz, prediction, n_label, dataset):
    """write a ply with colors for each class"""
    if len(prediction.shape) > 1 and prediction.shape[1] > 1:
        prediction = np.argmax(prediction, axis=1)
    color = np.zeros(xyz.shape)
    for i_label in range(0, n_label + 1):
        color[np.where(prediction == i_label), :] = get_color_from_label(i_label, dataset)
    prop = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    vertex_all = np.empty(len(xyz), dtype=prop)
    for i in range(0, 3):
        vertex_all[prop[i][0]] = xyz[:, i]
    for i in range(0, 3):
        vertex_all[prop[i + 3][0]] = color[:, i]
    ply = PlyData([PlyElement.describe(vertex_all, 'vertex')], text=True)
    ply.write(filename)

def prediction2ply_withlabels(filename, xyz, truth, prediction, n_label, dataset):
    """write a ply with colors for each class"""
    if len(prediction.shape) > 1 and prediction.shape[1] > 1:
        prediction = np.argmax(prediction, axis=1)
    color = np.zeros(xyz.shape)
    for i_label in range(0, n_label + 1):
        color[np.where(prediction == i_label), :] = get_color_from_label(i_label, dataset)
    prop = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('truth', 'i4'), ('pred', 'i4')]
    vertex_all = np.empty(len(xyz), dtype=prop)
    for i in range(0, 3):
        vertex_all[prop[i][0]] = xyz[:, i]
    for i in range(0, 3):
        vertex_all[prop[i + 3][0]] = color[:, i]
    vertex_all[prop[6][0]] = truth
    vertex_all[prop[7][0]] = prediction
    ply = PlyData([PlyElement.describe(vertex_all, 'vertex')], text=True)
    ply.write(filename)
# ------------------------------------------------------------------------------
def error2ply(filename, xyz, rgb, labels, prediction):
    """write a ply with green hue for correct classifcation and red for error"""
    if len(prediction.shape) > 1 and prediction.shape[1] > 1:
        prediction = np.argmax(prediction, axis=1)
    if len(labels.shape) > 1 and labels.shape[1] > 1:
        labels = np.argmax(labels, axis=1)
    color_rgb = rgb / 255
    for i_ver in range(0, len(labels)):

        color_hsv = list(colorsys.rgb_to_hsv(color_rgb[i_ver, 0], color_rgb[i_ver, 1], color_rgb[i_ver, 2]))
        if (labels[i_ver] == prediction[i_ver]) or (labels[i_ver] == 0):
            color_hsv[0] = 0.333333
        else:
            color_hsv[0] = 0
        color_hsv[1] = min(1, color_hsv[1] + 0.3)
        color_hsv[2] = min(1, color_hsv[2] + 0.1)
        color_rgb[i_ver, :] = list(colorsys.hsv_to_rgb(color_hsv[0], color_hsv[1], color_hsv[2]))
    color_rgb = np.array(color_rgb * 255, dtype='u1')
    prop = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    vertex_all = np.empty(len(xyz), dtype=prop)
    for i in range(0, 3):
        vertex_all[prop[i][0]] = xyz[:, i]
    for i in range(0, 3):
        vertex_all[prop[i + 3][0]] = color_rgb[:, i]
    ply = PlyData([PlyElement.describe(vertex_all, 'vertex')], text=True)
    ply.write(filename)


# ------------------------------------------------------------------------------
def spg2ply(filename, spg_graph):
    """write a ply displaying the SPG by adding edges between its centroid"""
    vertex_prop = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    vertex_val = np.empty((spg_graph['sp_centroids']).shape[0], dtype=vertex_prop)
    for i in range(0, 3):
        vertex_val[vertex_prop[i][0]] = spg_graph['sp_centroids'][:, i]
    edges_prop = [('vertex1', 'int32'), ('vertex2', 'int32')]
    edges_val = np.empty((spg_graph['source']).shape[0], dtype=edges_prop)
    edges_val[edges_prop[0][0]] = spg_graph['source'].flatten()
    edges_val[edges_prop[1][0]] = spg_graph['target'].flatten()
    ply = PlyData([PlyElement.describe(vertex_val, 'vertex'), PlyElement.describe(edges_val, 'edge')], text=True)
    ply.write(filename)


# ------------------------------------------------------------------------------
def scalar2ply(filename, xyz, scalar):
    """write a ply with an unisgned integer scalar field"""
    prop = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('scalar', 'f4')]
    vertex_all = np.empty(len(xyz), dtype=prop)
    for i in range(0, 3):
        vertex_all[prop[i][0]] = xyz[:, i]
    vertex_all[prop[3][0]] = scalar
    ply = PlyData([PlyElement.describe(vertex_all, 'vertex')], text=True)

    ply.write(filename)


# ------------------------------------------------------------------------------
def get_color_from_label(object_label, dataset):
    """associate the color corresponding to the class"""
    if dataset == 'custom_dataset':  # Custom set
        # object_label = {
        #     0: [0, 0, 0],  # unlabelled .->. black
        #     1: [170, 85, 0],  # 'ground' -> brown
        #     2: [0, 255, 0],  # 'vegetation' -> green
        #     3: [255, 255, 0],  # 'building' -> yellow
        #     4: [0, 255, 255],  # 'water' -> blue
        #     5: [255, 0, 255],  # 'vehicle'/'car' -> pink
        #     6: [0, 0, 153],  # 'boat' -> purple
        # }.get(object_label, -1)

        # object_label = {
        #     0: [0, 0, 0],  # unlabelled .->. black
        #     1: [170, 85, 0],  # 'ground' -> brown
        #     2: [0, 255, 0],  # 'vegetation' -> green
        #     3: [255, 255, 0],  # 'building' -> yellow
        #     4: [0, 255, 255],  # 'vehicle'/'car' -> pink
        # }.get(object_label, -1)

        # object_label = {
        #     0: [0, 0, 0],  # unlabelled .->. black
        #     1: [178,203,47],   # Low Vegetation
        #     2: [183,178,170],  # Impervious Surface
        #     3: [32,151,163],   # Vehicle
        #     4: [168,33,107],   # Urban Furniture
        #     5: [255,122,89],   # Roof
        #     6: [255,215,136],  # Facade
        #     7: [89,125,53],    # Shrub
        #     8: [0,128,65],     # Tree
        #     9: [170,85,0],     # Soil/Gravel
        #     10: [252,225,5],    # Vertical Surface
        #     11: [128,0,0],     # Chimney
        # }.get(object_label, -1)

        object_label = {
            0: [0, 0, 0],  # unlabelled .->. black
            1: [170, 85, 0],  # 'ground' -> brown
            2: [0, 255, 0],  # 'vegetation' -> green
            3: [255, 255, 0],  # 'building' -> yellow
            4: [0, 255, 255],  # 'water' -> blue
            5: [255, 0, 255],  # 'vehicle'/'car' -> pink
            6: [0, 0, 153],  # 'boat' -> purple
            7: [85, 85, 127],  # 'roof_surface'
            8: [255, 50, 50],  # 'chimney'
            9: [85, 0, 127],  # 'dormer'
            10: [50, 125, 150],  # 'balcony'
            11: [50, 0, 50],  # 'building_part'
            12: [215, 160, 140]  # 'wall'
        }.get(object_label, -1)
    else:
        raise ValueError('Unknown dataset: %s' % (dataset))
    if object_label == -1:
        raise ValueError('Type not recognized: %s' % (object_label))
    return object_label

# ------------------------------------------------------------------------------
def read_ply(filename):
    """convert from a ply file. include the label and the object number"""
    # ---read the ply file--------
    plydata = PlyData.read(filename)
    xyz = np.stack([plydata['vertex'][n] for n in ['x', 'y', 'z']], axis=1)
    try:
        rgb = np.stack([plydata['vertex'][n]
                        for n in ['red', 'green', 'blue']]
                       , axis=1).astype(np.uint8)
    except ValueError:
        rgb = np.stack([plydata['vertex'][n]
                        for n in ['r', 'g', 'b']]
                       , axis=1).astype(np.float32)
    if np.max(rgb) > 1:
        rgb = rgb
    try:
        #object_indices = plydata['vertex']['object_index']
        object_indices = plydata['vertex']['segment_id']
        labels = plydata['vertex']['label']
        return xyz, rgb, labels, object_indices
    except ValueError:
        try:
            labels = plydata['vertex']['label']
            return xyz, rgb, labels
        except ValueError:
            return xyz, rgb
        # try:
        #     labels = plydata['vertex']['label']
        #     rel_ele = plydata['vertex']['points_relative_ele']
        #     return xyz, rgb, labels, rel_ele
        # except ValueError:
        #     try:
        #         labels = plydata['vertex']['label']
        #         return xyz, rgb, labels
        #     except ValueError:
        #         return xyz, rgb

# ------------------------------------------------------------------------------
def read_graph_ply(filename):
    """convert from a ply file. include the label and the object number"""
    # ---read the ply file--------
    plydata = PlyData.read(filename)
    try:
        xyz = np.stack([plydata['vertex'][n] for n in ['x', 'y', 'z']], axis=1)
        edges = np.stack([plydata['edge'][n] for n in ['vertex1', 'vertex2']], axis=1)
        return xyz, edges
    except ValueError:
        return xyz

# ------------------------------------------------------------------------------
def write_ply_obj(filename, xyz, rgb, labels, object_indices):
    """write into a ply file. include the label and the object number"""
    prop = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1')
        , ('green', 'u1'), ('blue', 'u1'), ('label', 'u1')
        , ('object_index', 'uint32')]
    vertex_all = np.empty(len(xyz), dtype=prop)
    for i_prop in range(0, 3):
        vertex_all[prop[i_prop][0]] = xyz[:, i_prop]
    for i_prop in range(0, 3):
        vertex_all[prop[i_prop + 3][0]] = rgb[:, i_prop]
    vertex_all[prop[6][0]] = labels
    vertex_all[prop[7][0]] = object_indices
    ply = PlyData([PlyElement.describe(vertex_all, 'vertex')], text=True)
    ply.write(filename)

# ------------------------------------------------------------------------------
def embedding2ply(filename, xyz, embeddings):
    """write a ply with colors corresponding to geometric features"""

    if embeddings.shape[1] > 3:
        pca = PCA(n_components=3)
        # pca.fit(np.eye(embeddings.shape[1]))
        pca.fit(np.vstack((np.zeros((embeddings.shape[1],)), np.eye(embeddings.shape[1]))))
        embeddings = pca.transform(embeddings)

    # value = (embeddings-embeddings.mean(axis=0))/(2*embeddings.std())+0.5
    # value = np.minimum(np.maximum(value,0),1)
    # value = (embeddings)/(3 * embeddings.std())+0.5
    value = np.minimum(np.maximum((embeddings + 1) / 2, 0), 1)

    color = np.array(255 * value, dtype='uint8')
    prop = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    vertex_all = np.empty(len(xyz), dtype=prop)
    for i in range(0, 3):
        vertex_all[prop[i][0]] = xyz[:, i]
    for i in range(0, 3):
        vertex_all[prop[i + 3][0]] = color[:, i]
    ply = PlyData([PlyElement.describe(vertex_all, 'vertex')], text=True)

    ply.write(filename)

# ------------------------------------------------------------------------------
def edge_class2ply2(filename, edg_class, xyz, edg_source, edg_target):
    """write a ply with edge weight color coded into the midway point"""

    n_edg = len(edg_target)

    midpoint = (xyz[edg_source,] + xyz[edg_target,]) / 2

    color = np.zeros((edg_source.shape[0], 3), dtype='uint8')
    color[edg_class == 0,] = [0, 0, 0]
    color[(edg_class == 1).nonzero(),] = [255, 0, 0]
    color[(edg_class == 2).nonzero(),] = [125, 255, 0]
    color[(edg_class == 3).nonzero(),] = [0, 125, 255]

    prop = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    vertex_all = np.empty(n_edg, dtype=prop)
    for i in range(0, 3):
        vertex_all[prop[i][0]] = np.hstack(midpoint[:, i])
    for i in range(3, 6):
        vertex_all[prop[i][0]] = color[:, i - 3]

    ply = PlyData([PlyElement.describe(vertex_all, 'vertex')], text=True)

    ply.write(filename)

# ------------------------------------------------------------------------------
def write_ply_labels(filename, xyz, rgb, labels):
    """write into a ply file. include the label"""
    prop = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1')
        , ('blue', 'u1'), ('label', 'u1')]
    vertex_all = np.empty(len(xyz), dtype=prop)
    for i_prop in range(0, 3):
        vertex_all[prop[i_prop][0]] = xyz[:, i_prop]
    for i_prop in range(0, 3):
        vertex_all[prop[i_prop + 3][0]] = rgb[:, i_prop]
    vertex_all[prop[6][0]] = labels
    ply = PlyData([PlyElement.describe(vertex_all, 'vertex')], text=True)
    ply.write(filename)

# ------------------------------------------------------------------------------
def write_ply(filename, xyz, rgb):
    """write into a ply file"""
    prop = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    vertex_all = np.empty(len(xyz), dtype=prop)
    for i_prop in range(0, 3):
        vertex_all[prop[i_prop][0]] = xyz[:, i_prop]
    for i_prop in range(0, 3):
        vertex_all[prop[i_prop + 3][0]] = rgb[:, i_prop]
    ply = PlyData([PlyElement.describe(vertex_all, 'vertex')], text=True)
    ply.write(filename)


# ------------------------------------------------------------------------------
def write_features(file_name, geof, xyz, rgb, graph_nn, labels):
    """write the geometric features, labels and clouds in a h5 file"""
    if os.path.isfile(file_name):
        os.remove(file_name)
    data_file = h5py.File(file_name, 'w')
    data_file.create_dataset('geof', data=geof, dtype='float32')
    data_file.create_dataset('source', data=graph_nn["source"], dtype='uint32')
    data_file.create_dataset('target', data=graph_nn["target"], dtype='uint32')
    data_file.create_dataset('distances', data=graph_nn["distances"], dtype='float32')
    data_file.create_dataset('xyz', data=xyz, dtype='float32')
    if len(rgb) > 0:
        data_file.create_dataset('rgb', data=rgb, dtype='uint8')
    if len(labels) > 0 and len(labels.shape) > 1 and labels.shape[1] > 1:
        data_file.create_dataset('labels', data=labels, dtype='uint32')
    else:
        data_file.create_dataset('labels', data=labels, dtype='uint8')
    data_file.close()

def write_pssnet_features(file_name, geof, xyz, rgb, labels):
    """write the geometric features, labels and clouds in a h5 file"""
    if os.path.isfile(file_name):
        os.remove(file_name)
    data_file = h5py.File(file_name, 'w')
    data_file.create_dataset('geof', data=geof, dtype='float32')
    data_file.create_dataset('xyz', data=xyz, dtype='float32')
    if len(rgb) > 0:
        data_file.create_dataset('rgb', data=rgb, dtype='uint8')
    if len(labels) > 0 and len(labels.shape) > 1 and labels.shape[1] > 1:
        data_file.create_dataset('labels', data=labels, dtype='uint32')
    else:
        data_file.create_dataset('labels', data=labels, dtype='uint8')
    data_file.close()

# ------------------------------------------------------------------------------
def read_features(file_name):
    """read the geometric features, clouds and labels from a h5 file"""
    data_file = h5py.File(file_name, 'r')
    # fist get the number of vertices
    n_ver = len(data_file["geof"][:, 0])
    has_labels = len(data_file["labels"])
    # the labels can be empty in the case of a test set
    if has_labels:
        labels = np.array(data_file["labels"])
    else:
        labels = []
    # ---fill the arrays---
    geof = data_file["geof"][:]
    xyz = data_file["xyz"][:]
    rgb = data_file["rgb"][:]
    source = data_file["source"][:]
    target = data_file["target"][:]

    # ---set the graph---
    graph_nn = dict([("is_nn", True)])
    graph_nn["source"] = source
    graph_nn["target"] = target
    return geof, xyz, rgb, graph_nn, labels

def read_pssnet_features(file_name):
    """read the geometric features, clouds and labels from a h5 file"""
    data_file = h5py.File(file_name, 'r')
    # fist get the number of vertices
    n_ver = len(data_file["geof"][:, 0])
    has_labels = len(data_file["labels"])
    # the labels can be empty in the case of a test set
    if has_labels:
        labels = np.array(data_file["labels"])
    else:
        labels = []
    # ---fill the arrays---
    geof = data_file["geof"][:]
    xyz = data_file["xyz"][:]
    rgb = data_file["rgb"][:]

    return geof, xyz, rgb, labels
# ------------------------------------------------------------------------------
def write_pssnet_spg(file_name, graph_sp, components, in_component):
    """save the partition and spg information"""
    if os.path.isfile(file_name):
        os.remove(file_name)
    data_file = h5py.File(file_name, 'w')
    grp = data_file.create_group('components')
    n_com = len(components)
    for i_com in range(0, n_com):
        grp.create_dataset(str(i_com), data=components[i_com], dtype='uint32')
    data_file.create_dataset('in_component'
                             , data=in_component, dtype='uint32')
    data_file.create_dataset('sp_labels'
                             , data=graph_sp["sp_labels"], dtype='uint32')
    data_file.create_dataset('sp_centroids'
                             , data=graph_sp["sp_centroids"], dtype='float32')
    data_file.create_dataset('sp_linearity'
                             , data=graph_sp["sp_linearity"], dtype='float32')
    data_file.create_dataset('sp_verticality'
                             , data=graph_sp["sp_verticality"], dtype='float32')
    data_file.create_dataset('sp_curvature'
                             , data=graph_sp["sp_curvature"], dtype='float32')
    data_file.create_dataset('sp_sphericity'
                             , data=graph_sp["sp_sphericity"], dtype='float32')
    data_file.create_dataset('sp_planarity'
                             , data=graph_sp["sp_planarity"], dtype='float32')
    data_file.create_dataset('sp_vcount'
                             , data=graph_sp["sp_vcount"], dtype='float32')
    data_file.create_dataset('sp_triangle_density'
                             , data=graph_sp["sp_triangle_density"], dtype='float32')
    data_file.create_dataset('sp_inmat_rad'
                             , data=graph_sp["sp_inmat_rad"], dtype='float32')
    data_file.create_dataset('sp_shape_descriptor'
                             , data=graph_sp["sp_shape_descriptor"], dtype='float32')
    data_file.create_dataset('sp_compactness'
                             , data=graph_sp["sp_compactness"], dtype='float32')
    data_file.create_dataset('sp_shape_index'
                             , data=graph_sp["sp_shape_index"], dtype='float32')
    data_file.create_dataset('sp_pt2plane_dist_mean'
                             , data=graph_sp["sp_pt2plane_dist_mean"], dtype='float32')
    data_file.create_dataset('sp_point_red'
                             , data=graph_sp["sp_point_red"], dtype='float32')
    data_file.create_dataset('sp_point_green'
                             , data=graph_sp["sp_point_green"], dtype='float32')
    data_file.create_dataset('sp_point_blue'
                             , data=graph_sp["sp_point_blue"], dtype='float32')
    data_file.create_dataset('sp_point_hue'
                             , data=graph_sp["sp_point_hue"], dtype='float32')
    data_file.create_dataset('sp_point_sat'
                             , data=graph_sp["sp_point_sat"], dtype='float32')
    data_file.create_dataset('sp_point_val'
                             , data=graph_sp["sp_point_val"], dtype='float32')
    data_file.create_dataset('sp_point_hue_var'
                             , data=graph_sp["sp_point_hue_var"], dtype='float32')
    data_file.create_dataset('sp_point_sat_var'
                             , data=graph_sp["sp_point_sat_var"], dtype='float32')
    data_file.create_dataset('sp_point_val_var'
                             , data=graph_sp["sp_point_val_var"], dtype='float32')
    data_file.create_dataset('sp_point_greenness'
                             , data=graph_sp["sp_point_greenness"], dtype='float32')
    data_file.create_dataset('sp_point_hue_bin_0'
                             , data=graph_sp["sp_point_hue_bin_0"], dtype='float32')
    data_file.create_dataset('sp_point_hue_bin_1'
                             , data=graph_sp["sp_point_hue_bin_1"], dtype='float32')
    data_file.create_dataset('sp_point_hue_bin_2'
                             , data=graph_sp["sp_point_hue_bin_2"], dtype='float32')
    data_file.create_dataset('sp_point_hue_bin_3'
                             , data=graph_sp["sp_point_hue_bin_3"], dtype='float32')
    data_file.create_dataset('sp_point_hue_bin_4'
                             , data=graph_sp["sp_point_hue_bin_4"], dtype='float32')
    data_file.create_dataset('sp_point_hue_bin_5'
                             , data=graph_sp["sp_point_hue_bin_5"], dtype='float32')
    data_file.create_dataset('sp_point_hue_bin_6'
                             , data=graph_sp["sp_point_hue_bin_6"], dtype='float32')
    data_file.create_dataset('sp_point_hue_bin_7'
                             , data=graph_sp["sp_point_hue_bin_7"], dtype='float32')
    data_file.create_dataset('sp_point_hue_bin_8'
                             , data=graph_sp["sp_point_hue_bin_8"], dtype='float32')
    data_file.create_dataset('sp_point_hue_bin_9'
                             , data=graph_sp["sp_point_hue_bin_9"], dtype='float32')
    data_file.create_dataset('sp_point_hue_bin_10'
                             , data=graph_sp["sp_point_hue_bin_10"], dtype='float32')
    data_file.create_dataset('sp_point_hue_bin_11'
                             , data=graph_sp["sp_point_hue_bin_11"], dtype='float32')
    data_file.create_dataset('sp_point_hue_bin_12'
                             , data=graph_sp["sp_point_hue_bin_12"], dtype='float32')
    data_file.create_dataset('sp_point_hue_bin_13'
                             , data=graph_sp["sp_point_hue_bin_13"], dtype='float32')
    data_file.create_dataset('sp_point_hue_bin_14'
                             , data=graph_sp["sp_point_hue_bin_14"], dtype='float32')
    data_file.create_dataset('sp_point_sat_bin_0'
                             , data=graph_sp["sp_point_sat_bin_0"], dtype='float32')
    data_file.create_dataset('sp_point_sat_bin_1'
                             , data=graph_sp["sp_point_sat_bin_1"], dtype='float32')
    data_file.create_dataset('sp_point_sat_bin_2'
                             , data=graph_sp["sp_point_sat_bin_2"], dtype='float32')
    data_file.create_dataset('sp_point_sat_bin_3'
                             , data=graph_sp["sp_point_sat_bin_3"], dtype='float32')
    data_file.create_dataset('sp_point_sat_bin_4'
                             , data=graph_sp["sp_point_sat_bin_4"], dtype='float32')
    data_file.create_dataset('sp_point_val_bin_0'
                             , data=graph_sp["sp_point_val_bin_0"], dtype='float32')
    data_file.create_dataset('sp_point_val_bin_1'
                             , data=graph_sp["sp_point_val_bin_1"], dtype='float32')
    data_file.create_dataset('sp_point_val_bin_2'
                             , data=graph_sp["sp_point_val_bin_2"], dtype='float32')
    data_file.create_dataset('sp_point_val_bin_3'
                             , data=graph_sp["sp_point_val_bin_3"], dtype='float32')
    data_file.create_dataset('sp_point_val_bin_4'
                             , data=graph_sp["sp_point_val_bin_4"], dtype='float32')
    data_file.create_dataset('source'
                             , data=graph_sp["source"], dtype='uint32')
    data_file.create_dataset('target'
                             , data=graph_sp["target"], dtype='uint32')
    data_file.create_dataset('se_delta_mean'
                             , data=graph_sp["se_delta_mean"], dtype='float32')
    data_file.create_dataset('se_delta_std'
                             , data=graph_sp["se_delta_std"], dtype='float32')
# -----------------------------------------------------------------------------
def read_pssnet_spg(file_name):
    """read the partition and spg information"""
    data_file = h5py.File(file_name, 'r')
    graph = dict([("is_nn", False)])
    graph["source"] = np.array(data_file["source"], dtype='uint32')
    graph["target"] = np.array(data_file["target"], dtype='uint32')
    graph["sp_centroids"] = np.array(data_file["sp_centroids"], dtype='float32')
    graph["sp_linearity"] = np.array(data_file["sp_linearity"], dtype='float32')
    graph["sp_verticality"] = np.array(data_file["sp_verticality"], dtype='float32')
    graph["sp_curvature"] = np.array(data_file["sp_curvature"], dtype='float32')
    graph["sp_sphericity"] = np.array(data_file["sp_sphericity"], dtype='float32')
    graph["sp_planarity"] = np.array(data_file["sp_planarity"], dtype='float32')

    graph["sp_vcount"] = np.array(data_file["sp_vcount"], dtype='float32')
    graph["sp_triangle_density"] = np.array(data_file["sp_triangle_density"], dtype='float32')
    graph["sp_inmat_rad"] = np.array(data_file["sp_inmat_rad"], dtype='float32')
    graph["sp_shape_descriptor"] = np.array(data_file["sp_shape_descriptor"], dtype='float32')
    graph["sp_compactness"] = np.array(data_file["sp_compactness"], dtype='float32')
    graph["sp_shape_index"] = np.array(data_file["sp_shape_index"], dtype='float32')
    graph["sp_pt2plane_dist_mean"] = np.array(data_file["sp_pt2plane_dist_mean"], dtype='float32')

    graph["sp_point_red"] = np.array(data_file["sp_point_red"], dtype='float32')
    graph["sp_point_green"] = np.array(data_file["sp_point_green"], dtype='float32')
    graph["sp_point_blue"] = np.array(data_file["sp_point_blue"], dtype='float32')
    graph["sp_point_hue"] = np.array(data_file["sp_point_hue"], dtype='float32')
    graph["sp_point_sat"] = np.array(data_file["sp_point_sat"], dtype='float32')
    graph["sp_point_val"] = np.array(data_file["sp_point_val"], dtype='float32')
    graph["sp_point_hue_var"] = np.array(data_file["sp_point_hue_var"], dtype='float32')
    graph["sp_point_sat_var"] = np.array(data_file["sp_point_sat_var"], dtype='float32')
    graph["sp_point_val_var"] = np.array(data_file["sp_point_val_var"], dtype='float32')
    graph["sp_point_greenness"] = np.array(data_file["sp_point_greenness"], dtype='float32')
    graph["sp_point_hue_bin_0"] = np.array(data_file["sp_point_hue_bin_0"], dtype='float32')
    graph["sp_point_hue_bin_1"] = np.array(data_file["sp_point_hue_bin_1"], dtype='float32')
    graph["sp_point_hue_bin_2"] = np.array(data_file["sp_point_hue_bin_2"], dtype='float32')
    graph["sp_point_hue_bin_3"] = np.array(data_file["sp_point_hue_bin_3"], dtype='float32')
    graph["sp_point_hue_bin_4"] = np.array(data_file["sp_point_hue_bin_4"], dtype='float32')
    graph["sp_point_hue_bin_5"] = np.array(data_file["sp_point_hue_bin_5"], dtype='float32')
    graph["sp_point_hue_bin_6"] = np.array(data_file["sp_point_hue_bin_6"], dtype='float32')
    graph["sp_point_hue_bin_7"] = np.array(data_file["sp_point_hue_bin_7"], dtype='float32')
    graph["sp_point_hue_bin_8"] = np.array(data_file["sp_point_hue_bin_8"], dtype='float32')
    graph["sp_point_hue_bin_9"] = np.array(data_file["sp_point_hue_bin_9"], dtype='float32')
    graph["sp_point_hue_bin_10"] = np.array(data_file["sp_point_hue_bin_10"], dtype='float32')
    graph["sp_point_hue_bin_11"] = np.array(data_file["sp_point_hue_bin_11"], dtype='float32')
    graph["sp_point_hue_bin_12"] = np.array(data_file["sp_point_hue_bin_12"], dtype='float32')
    graph["sp_point_hue_bin_13"] = np.array(data_file["sp_point_hue_bin_13"], dtype='float32')
    graph["sp_point_hue_bin_14"] = np.array(data_file["sp_point_hue_bin_14"], dtype='float32')
    graph["sp_point_sat_bin_0"] = np.array(data_file["sp_point_sat_bin_0"], dtype='float32')
    graph["sp_point_sat_bin_1"] = np.array(data_file["sp_point_sat_bin_1"], dtype='float32')
    graph["sp_point_sat_bin_2"] = np.array(data_file["sp_point_sat_bin_2"], dtype='float32')
    graph["sp_point_sat_bin_3"] = np.array(data_file["sp_point_sat_bin_3"], dtype='float32')
    graph["sp_point_sat_bin_4"] = np.array(data_file["sp_point_sat_bin_4"], dtype='float32')
    graph["sp_point_val_bin_0"] = np.array(data_file["sp_point_val_bin_0"], dtype='float32')
    graph["sp_point_val_bin_1"] = np.array(data_file["sp_point_val_bin_1"], dtype='float32')
    graph["sp_point_val_bin_2"] = np.array(data_file["sp_point_val_bin_2"], dtype='float32')
    graph["sp_point_val_bin_3"] = np.array(data_file["sp_point_val_bin_3"], dtype='float32')
    graph["sp_point_val_bin_4"] = np.array(data_file["sp_point_val_bin_4"], dtype='float32')
    graph["se_delta_mean"] = np.array(data_file["se_delta_mean"], dtype='float32')
    graph["se_delta_std"] = np.array(data_file["se_delta_std"], dtype='float32')
    in_component = np.array(data_file["in_component"], dtype='uint32')
    n_com = len(graph["sp_verticality"])
    graph["sp_labels"] = np.array(data_file["sp_labels"], dtype='uint32')
    grp = data_file['components']
    components = np.empty((n_com,), dtype=object)
    for i_com in range(0, n_com):
        components[i_com] = np.array(grp[str(i_com)], dtype='uint32').tolist()
    return graph, components, in_component
# ------------------------------------------------------------------------------
def reduced_labels2full(labels_red, components, n_ver):
    """distribute the labels of superpoints to their repsective points"""
    labels_full = np.zeros((n_ver,), dtype='uint8')
    for i_com in range(0, len(components)):
        labels_full[components[i_com]] = labels_red[i_com]
    return labels_full

# ------------------------------------------------------------------------------
def interpolate_labels_batch(data_file, xyz, labels, ver_batch):
    """interpolate the labels of the pruned cloud to the full cloud"""
    if len(labels.shape) > 1 and labels.shape[1] > 1:
        labels = np.argmax(labels, axis=1)
    i_rows = 0
    labels_f = np.zeros((0,), dtype='uint8')
    # ---the clouds can potentially be too big to parse directly---
    # ---they are cut in batches in the order they are stored---
    nn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(xyz)
    while True:
        try:
            if ver_batch > 0:
                if i_rows is None:
                    print("read lines %d to %d" % (0, ver_batch))
                else:
                    print("read lines %d to %d" % (i_rows, i_rows + ver_batch))
                # vertices = np.genfromtxt(data_file
                #         , delimiter=' ', max_rows=ver_batch
                #        , skip_header=i_rows)
                vertices = pd.read_csv(data_file
                                       , sep=' ', nrows=ver_batch
                                       , header=i_rows).values
            else:
                # vertices = np.genfromtxt(data_file
                #        , delimiter=' ')
                vertices = pd.read_csv(data_file
                                       , delimiter=' ').values
                break
        except (StopIteration, pd.errors.ParserError):
            # end of file
            break
        if len(vertices) == 0:
            break
        xyz_full = np.array(vertices[:, 0:3], dtype='float32')
        del vertices
        distances, neighbor = nn.kneighbors(xyz_full)
        del distances
        labels_f = np.hstack((labels_f, labels[neighbor].flatten()))
        i_rows = i_rows + ver_batch
    return labels_f


# ------------------------------------------------------------------------------
def interpolate_labels(xyz_up, xyz, labels, ver_batch):
    """interpolate the labels of the pruned cloud to the full cloud"""
    if len(labels.shape) > 1 and labels.shape[1] > 1:
        labels = np.argmax(labels, axis=1)
    nn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(xyz)
    distances, neighbor = nn.kneighbors(xyz_up)
    return labels[neighbor].flatten()


# ------------------------------------------------------------------------------
def perfect_prediction(components, labels):
    """assign each superpoint with the majority label"""
    #full_pred = np.zeros((labels.shape[0],), dtype='uint32')
    full_pred = np.zeros((labels.shape[1],), dtype='uint32')
    for i_com in range(len(components)):
        #label_com = labels[components[i_com], 1:].sum(0).argmax()
        label_com = np.argmax(np.bincount(labels[0, components[i_com]]))
        full_pred[components[i_com]] = label_com
    return full_pred

