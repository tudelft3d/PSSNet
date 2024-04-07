# ------------------------------------------------------------------------------
# ---------  Graph methods for PSSNet   ------------------------------
# ---------    Weixiao GAO, Feb. 2023     -----------------------------------
# ------------------------------------------------------------------------------
import progressbar
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import Delaunay
from numpy import linalg as LA
import numpy.matlib


# ------------------------------------------------------------------------------
def compute_graph_nn(xyz, k_nn):
    """compute the knn graph"""
    num_ver = xyz.shape[0]
    graph = dict([("is_nn", True)])
    nn = NearestNeighbors(n_neighbors=k_nn + 1, algorithm='kd_tree').fit(xyz)
    distances, neighbors = nn.kneighbors(xyz)
    neighbors = neighbors[:, 1:]
    distances = distances[:, 1:]
    source = np.matlib.repmat(range(0, num_ver), k_nn, 1).flatten(order='F')
    # save the graph
    graph["source"] = source.flatten().astype('uint32')
    graph["target"] = neighbors.flatten().astype('uint32')
    graph["distances"] = distances.flatten().astype('float32')
    return graph


# ------------------------------------------------------------------------------
def compute_graph_nn_2(xyz, k_nn1, k_nn2, voronoi=0.0):
    """compute simulteneoulsy 2 knn structures
    only saves target for knn2
    assumption : knn1 <= knn2"""
    assert k_nn1 <= k_nn2, "knn1 must be smaller than knn2"
    n_ver = xyz.shape[0]
    # compute nearest neighbors
    graph = dict([("is_nn", True)])
    nn = NearestNeighbors(n_neighbors=k_nn2 + 1, algorithm='kd_tree').fit(xyz)
    distances, neighbors = nn.kneighbors(xyz)
    del nn
    neighbors = neighbors[:, 1:]
    distances = distances[:, 1:]
    # ---knn2---
    target2 = (neighbors.flatten()).astype('uint32')
    # ---knn1-----
    if voronoi > 0:
        tri = Delaunay(xyz)
        graph["source"] = np.hstack((tri.vertices[:, 0], tri.vertices[:, 0], \
                                     tri.vertices[:, 0], tri.vertices[:, 1], tri.vertices[:, 1],
                                     tri.vertices[:, 2])).astype('uint64')
        graph["target"] = np.hstack((tri.vertices[:, 1], tri.vertices[:, 2], \
                                     tri.vertices[:, 3], tri.vertices[:, 2], tri.vertices[:, 3],
                                     tri.vertices[:, 3])).astype('uint64')
        graph["distances"] = ((xyz[graph["source"], :] - xyz[graph["target"], :]) ** 2).sum(1)
        keep_edges = graph["distances"] < voronoi
        graph["source"] = graph["source"][keep_edges]
        graph["target"] = graph["target"][keep_edges]

        graph["source"] = np.hstack((graph["source"], np.matlib.repmat(range(0, n_ver)
                                                                       , k_nn1, 1).flatten(order='F').astype('uint32')))
        neighbors = neighbors[:, :k_nn1]
        graph["target"] = np.hstack((graph["target"], np.transpose(neighbors.flatten(order='C')).astype('uint32')))

        edg_id = graph["source"] + n_ver * graph["target"]

        dump, unique_edges = np.unique(edg_id, return_index=True)
        graph["source"] = graph["source"][unique_edges]
        graph["target"] = graph["target"][unique_edges]

        graph["distances"] = graph["distances"][keep_edges]
    else:
        neighbors = neighbors[:, :k_nn1]
        distances = distances[:, :k_nn1]
        graph["source"] = np.matlib.repmat(range(0, n_ver)
                                           , k_nn1, 1).flatten(order='F').astype('uint32')
        graph["target"] = np.transpose(neighbors.flatten(order='C')).astype('uint32')
        graph["distances"] = distances.flatten().astype('float32')
    # save the graph
    return graph, target2


# ------------------------------------------------------------------------------
def compute_pssnet_sp_graph(xyz, d_max, in_component, components, pssnetg_edges, labels, n_labels, fea_com):
    """compute the superpoint graph with superpoints and superedges features"""
    n_com = max(in_component) + 1
    in_component = np.array(in_component)
    has_labels = len(labels) > 1
    label_hist = has_labels and len(labels.shape) > 1 and labels.shape[1] > 1

    print("    Add edges from input graph ...")
    pssnetg_edges = np.unique(pssnetg_edges, axis=1)
    edges_added = np.transpose(pssnetg_edges)
    del pssnetg_edges

    if d_max > 0:
        dist = np.sqrt(((xyz[edges_added[0, :]] - xyz[edges_added[1, :]]) ** 2).sum(1))
        edges_added = edges_added[:, dist < d_max]

    # ---sort edges by alpha numeric order wrt to the components of their source/target---
    # use delaunay + additional edges
    print("    Sort edges ...")
    n_edg = len(edges_added[0])
    edge_comp = in_component[edges_added]
    edge_comp_index = n_com * edge_comp[0, :] + edge_comp[1, :]
    order = np.argsort(edge_comp_index)
    edges_added = edges_added[:, order]
    edge_comp = edge_comp[:, order]
    edge_comp_index = edge_comp_index[order]
    # marks where the edges change components iot compting them by blocks
    jump_edg = np.vstack((0, np.argwhere(np.diff(edge_comp_index)) + 1, n_edg)).flatten()
    n_sedg = len(jump_edg) - 1

    # ---set up the edges descriptors---
    print("    Set up the edges descriptor ...")
    graph = dict([("is_nn", False)])
    # ---Common features---
    graph["sp_centroids"] = np.zeros((n_com, 3), dtype='float32')
    graph["se_delta_mean"] = np.zeros((n_sedg, 3), dtype='float32')
    graph["se_delta_std"] = np.zeros((n_sedg, 3), dtype='float32')

    graph["sp_linearity"] = np.zeros((n_com, 1), dtype='float32')
    graph["sp_verticality"] = np.zeros((n_com, 1), dtype='float32')
    graph["sp_curvature"] = np.zeros((n_com, 1), dtype='float32')
    graph["sp_sphericity"] = np.zeros((n_com, 1), dtype='float32')
    graph["sp_planarity"] = np.zeros((n_com, 1), dtype='float32')

    #Shape features
    graph["sp_vcount"] = np.zeros((n_com, 1), dtype='float32')
    graph["sp_triangle_density"] = np.zeros((n_com, 1), dtype='float32')
    graph["sp_inmat_rad"] = np.zeros((n_com, 1), dtype='float32')
    graph["sp_shape_descriptor"] = np.zeros((n_com, 1), dtype='float32')
    graph["sp_compactness"] = np.zeros((n_com, 1), dtype='float32')
    graph["sp_shape_index"] = np.zeros((n_com, 1), dtype='float32')
    graph["sp_pt2plane_dist_mean"] = np.zeros((n_com, 1), dtype='float32')

    #Color features
    graph["sp_point_red"] = np.zeros((n_com, 1), dtype='float32')
    graph["sp_point_green"] = np.zeros((n_com, 1), dtype='float32')
    graph["sp_point_blue"] = np.zeros((n_com, 1), dtype='float32')
    graph["sp_point_hue"] = np.zeros((n_com, 1), dtype='float32')
    graph["sp_point_sat"] = np.zeros((n_com, 1), dtype='float32')
    graph["sp_point_val"] = np.zeros((n_com, 1), dtype='float32')
    graph["sp_point_hue_var"] = np.zeros((n_com, 1), dtype='float32')
    graph["sp_point_sat_var"] = np.zeros((n_com, 1), dtype='float32')
    graph["sp_point_val_var"] = np.zeros((n_com, 1), dtype='float32')
    graph["sp_point_greenness"] = np.zeros((n_com, 1), dtype='float32')

    graph["sp_point_hue_bin_0"] = np.zeros((n_com, 1), dtype='float32')
    graph["sp_point_hue_bin_1"] = np.zeros((n_com, 1), dtype='float32')
    graph["sp_point_hue_bin_2"] = np.zeros((n_com, 1), dtype='float32')
    graph["sp_point_hue_bin_3"] = np.zeros((n_com, 1), dtype='float32')
    graph["sp_point_hue_bin_4"] = np.zeros((n_com, 1), dtype='float32')
    graph["sp_point_hue_bin_5"] = np.zeros((n_com, 1), dtype='float32')
    graph["sp_point_hue_bin_6"] = np.zeros((n_com, 1), dtype='float32')
    graph["sp_point_hue_bin_7"] = np.zeros((n_com, 1), dtype='float32')
    graph["sp_point_hue_bin_8"] = np.zeros((n_com, 1), dtype='float32')
    graph["sp_point_hue_bin_9"] = np.zeros((n_com, 1), dtype='float32')
    graph["sp_point_hue_bin_10"] = np.zeros((n_com, 1), dtype='float32')
    graph["sp_point_hue_bin_11"] = np.zeros((n_com, 1), dtype='float32')
    graph["sp_point_hue_bin_12"] = np.zeros((n_com, 1), dtype='float32')
    graph["sp_point_hue_bin_13"] = np.zeros((n_com, 1), dtype='float32')
    graph["sp_point_hue_bin_14"] = np.zeros((n_com, 1), dtype='float32')

    graph["sp_point_sat_bin_0"] = np.zeros((n_com, 1), dtype='float32')
    graph["sp_point_sat_bin_1"] = np.zeros((n_com, 1), dtype='float32')
    graph["sp_point_sat_bin_2"] = np.zeros((n_com, 1), dtype='float32')
    graph["sp_point_sat_bin_3"] = np.zeros((n_com, 1), dtype='float32')
    graph["sp_point_sat_bin_4"] = np.zeros((n_com, 1), dtype='float32')

    graph["sp_point_val_bin_0"] = np.zeros((n_com, 1), dtype='float32')
    graph["sp_point_val_bin_1"] = np.zeros((n_com, 1), dtype='float32')
    graph["sp_point_val_bin_2"] = np.zeros((n_com, 1), dtype='float32')
    graph["sp_point_val_bin_3"] = np.zeros((n_com, 1), dtype='float32')
    graph["sp_point_val_bin_4"] = np.zeros((n_com, 1), dtype='float32')

    graph["source"] = np.zeros((n_sedg, 1), dtype='uint32')
    graph["target"] = np.zeros((n_sedg, 1), dtype='uint32')
    if has_labels:
        graph["sp_labels"] = np.zeros((n_com, n_labels + 1), dtype='uint32')
    else:
        graph["sp_labels"] = []
    # ---compute the superpoint features---
    for i_com in range(0, n_com):
        comp = components[i_com]
        if has_labels and not label_hist:
            graph["sp_labels"][i_com, :] = np.histogram(labels[comp]
                                                        , bins=[float(i) - 0.5 for i in range(0, n_labels + 2)])[0]
        if has_labels and label_hist:
            graph["sp_labels"][i_com, :] = sum(labels[comp, :])
        xyz_sp = np.unique(xyz[comp, :], axis=0)
        if len(xyz_sp) == 1:
            graph["sp_centroids"][i_com] = xyz_sp

            # Eigen features
            graph["sp_linearity"][i_com] = fea_com[i_com, 0]
            graph["sp_verticality"][i_com] = fea_com[i_com, 1]
            graph["sp_curvature"][i_com] = fea_com[i_com, 2]
            graph["sp_sphericity"][i_com] = fea_com[i_com, 3]
            graph["sp_planarity"][i_com] = fea_com[i_com, 4]

            # Shape features
            graph["sp_vcount"][i_com] = fea_com[i_com, 5]
            graph["sp_triangle_density"][i_com] = fea_com[i_com, 6]
            graph["sp_inmat_rad"][i_com] = fea_com[i_com, 7]
            graph["sp_shape_descriptor"][i_com] = fea_com[i_com, 8]
            graph["sp_compactness"][i_com] = fea_com[i_com, 9]
            graph["sp_shape_index"][i_com] = fea_com[i_com, 10]
            graph["sp_pt2plane_dist_mean"][i_com] = fea_com[i_com, 11]

            # Color features
            graph["sp_point_red"][i_com] = fea_com[i_com, 12]
            graph["sp_point_green"][i_com] = fea_com[i_com, 13]
            graph["sp_point_blue"][i_com] = fea_com[i_com, 14]
            graph["sp_point_hue"][i_com] = fea_com[i_com, 15]
            graph["sp_point_sat"][i_com] = fea_com[i_com, 16]
            graph["sp_point_val"][i_com] = fea_com[i_com, 17]
            graph["sp_point_hue_var"][i_com] = fea_com[i_com, 18]
            graph["sp_point_sat_var"][i_com] = fea_com[i_com, 19]
            graph["sp_point_val_var"][i_com] = fea_com[i_com, 20]
            graph["sp_point_greenness"][i_com] = fea_com[i_com, 21]

            graph["sp_point_hue_bin_0"][i_com] = fea_com[i_com, 22]
            graph["sp_point_hue_bin_1"][i_com] = fea_com[i_com, 23]
            graph["sp_point_hue_bin_2"][i_com] = fea_com[i_com, 24]
            graph["sp_point_hue_bin_3"][i_com] = fea_com[i_com, 25]
            graph["sp_point_hue_bin_4"][i_com] = fea_com[i_com, 26]
            graph["sp_point_hue_bin_5"][i_com] = fea_com[i_com, 27]
            graph["sp_point_hue_bin_6"][i_com] = fea_com[i_com, 28]
            graph["sp_point_hue_bin_7"][i_com] = fea_com[i_com, 29]
            graph["sp_point_hue_bin_8"][i_com] = fea_com[i_com, 30]
            graph["sp_point_hue_bin_9"][i_com] = fea_com[i_com, 31]
            graph["sp_point_hue_bin_10"][i_com] = fea_com[i_com, 32]
            graph["sp_point_hue_bin_11"][i_com] = fea_com[i_com, 33]
            graph["sp_point_hue_bin_12"][i_com] = fea_com[i_com, 34]
            graph["sp_point_hue_bin_13"][i_com] = fea_com[i_com, 35]
            graph["sp_point_hue_bin_14"][i_com] = fea_com[i_com, 36]

            graph["sp_point_sat_bin_0"][i_com] = fea_com[i_com, 37]
            graph["sp_point_sat_bin_1"][i_com] = fea_com[i_com, 38]
            graph["sp_point_sat_bin_2"][i_com] = fea_com[i_com, 39]
            graph["sp_point_sat_bin_3"][i_com] = fea_com[i_com, 40]
            graph["sp_point_sat_bin_4"][i_com] = fea_com[i_com, 41]

            graph["sp_point_val_bin_0"][i_com] = fea_com[i_com, 42]
            graph["sp_point_val_bin_1"][i_com] = fea_com[i_com, 43]
            graph["sp_point_val_bin_2"][i_com] = fea_com[i_com, 44]
            graph["sp_point_val_bin_3"][i_com] = fea_com[i_com, 45]
            graph["sp_point_val_bin_4"][i_com] = fea_com[i_com, 46]

        elif len(xyz_sp) == 2:
            graph["sp_centroids"][i_com] = np.mean(xyz_sp, axis=0)

            # Eigen features
            graph["sp_linearity"][i_com] = fea_com[i_com, 0]
            graph["sp_verticality"][i_com] = fea_com[i_com, 1]
            graph["sp_curvature"][i_com] = fea_com[i_com, 2]
            graph["sp_sphericity"][i_com] = fea_com[i_com, 3]
            graph["sp_planarity"][i_com] = fea_com[i_com, 4]

            # Shape features
            graph["sp_vcount"][i_com] = fea_com[i_com, 5]
            graph["sp_triangle_density"][i_com] = fea_com[i_com, 6]
            graph["sp_inmat_rad"][i_com] = fea_com[i_com, 7]
            graph["sp_shape_descriptor"][i_com] = fea_com[i_com, 8]
            graph["sp_compactness"][i_com] = fea_com[i_com, 9]
            graph["sp_shape_index"][i_com] = fea_com[i_com, 10]
            graph["sp_pt2plane_dist_mean"][i_com] = fea_com[i_com, 11]

            # Color features
            graph["sp_point_red"][i_com] = fea_com[i_com, 12]
            graph["sp_point_green"][i_com] = fea_com[i_com, 13]
            graph["sp_point_blue"][i_com] = fea_com[i_com, 14]
            graph["sp_point_hue"][i_com] = fea_com[i_com, 15]
            graph["sp_point_sat"][i_com] = fea_com[i_com, 16]
            graph["sp_point_val"][i_com] = fea_com[i_com, 17]
            graph["sp_point_hue_var"][i_com] = fea_com[i_com, 18]
            graph["sp_point_sat_var"][i_com] = fea_com[i_com, 19]
            graph["sp_point_val_var"][i_com] = fea_com[i_com, 20]
            graph["sp_point_greenness"][i_com] = fea_com[i_com, 21]

            graph["sp_point_hue_bin_0"][i_com] = fea_com[i_com, 22]
            graph["sp_point_hue_bin_1"][i_com] = fea_com[i_com, 23]
            graph["sp_point_hue_bin_2"][i_com] = fea_com[i_com, 24]
            graph["sp_point_hue_bin_3"][i_com] = fea_com[i_com, 25]
            graph["sp_point_hue_bin_4"][i_com] = fea_com[i_com, 26]
            graph["sp_point_hue_bin_5"][i_com] = fea_com[i_com, 27]
            graph["sp_point_hue_bin_6"][i_com] = fea_com[i_com, 28]
            graph["sp_point_hue_bin_7"][i_com] = fea_com[i_com, 29]
            graph["sp_point_hue_bin_8"][i_com] = fea_com[i_com, 30]
            graph["sp_point_hue_bin_9"][i_com] = fea_com[i_com, 31]
            graph["sp_point_hue_bin_10"][i_com] = fea_com[i_com, 32]
            graph["sp_point_hue_bin_11"][i_com] = fea_com[i_com, 33]
            graph["sp_point_hue_bin_12"][i_com] = fea_com[i_com, 34]
            graph["sp_point_hue_bin_13"][i_com] = fea_com[i_com, 35]
            graph["sp_point_hue_bin_14"][i_com] = fea_com[i_com, 36]

            graph["sp_point_sat_bin_0"][i_com] = fea_com[i_com, 37]
            graph["sp_point_sat_bin_1"][i_com] = fea_com[i_com, 38]
            graph["sp_point_sat_bin_2"][i_com] = fea_com[i_com, 39]
            graph["sp_point_sat_bin_3"][i_com] = fea_com[i_com, 40]
            graph["sp_point_sat_bin_4"][i_com] = fea_com[i_com, 41]

            graph["sp_point_val_bin_0"][i_com] = fea_com[i_com, 42]
            graph["sp_point_val_bin_1"][i_com] = fea_com[i_com, 43]
            graph["sp_point_val_bin_2"][i_com] = fea_com[i_com, 44]
            graph["sp_point_val_bin_3"][i_com] = fea_com[i_com, 45]
            graph["sp_point_val_bin_4"][i_com] = fea_com[i_com, 46]
        else:
            graph["sp_centroids"][i_com] = np.mean(xyz_sp, axis=0)

            # Eigen features
            graph["sp_linearity"][i_com] = fea_com[i_com, 0]
            graph["sp_verticality"][i_com] = fea_com[i_com, 1]
            graph["sp_curvature"][i_com] = fea_com[i_com, 2]
            graph["sp_sphericity"][i_com] = fea_com[i_com, 3]
            graph["sp_planarity"][i_com] = fea_com[i_com, 4]

            # Shape features
            graph["sp_vcount"][i_com] = fea_com[i_com, 5]
            graph["sp_triangle_density"][i_com] = fea_com[i_com, 6]
            graph["sp_inmat_rad"][i_com] = fea_com[i_com, 7]
            graph["sp_shape_descriptor"][i_com] = fea_com[i_com, 8]
            graph["sp_compactness"][i_com] = fea_com[i_com, 9]
            graph["sp_shape_index"][i_com] = fea_com[i_com, 10]
            graph["sp_pt2plane_dist_mean"][i_com] = fea_com[i_com, 11]

            # Color features
            graph["sp_point_red"][i_com] = fea_com[i_com, 12]
            graph["sp_point_green"][i_com] = fea_com[i_com, 13]
            graph["sp_point_blue"][i_com] = fea_com[i_com, 14]
            graph["sp_point_hue"][i_com] = fea_com[i_com, 15]
            graph["sp_point_sat"][i_com] = fea_com[i_com, 16]
            graph["sp_point_val"][i_com] = fea_com[i_com, 17]
            graph["sp_point_hue_var"][i_com] = fea_com[i_com, 18]
            graph["sp_point_sat_var"][i_com] = fea_com[i_com, 19]
            graph["sp_point_val_var"][i_com] = fea_com[i_com, 20]
            graph["sp_point_greenness"][i_com] = fea_com[i_com, 21]

            graph["sp_point_hue_bin_0"][i_com] = fea_com[i_com, 22]
            graph["sp_point_hue_bin_1"][i_com] = fea_com[i_com, 23]
            graph["sp_point_hue_bin_2"][i_com] = fea_com[i_com, 24]
            graph["sp_point_hue_bin_3"][i_com] = fea_com[i_com, 25]
            graph["sp_point_hue_bin_4"][i_com] = fea_com[i_com, 26]
            graph["sp_point_hue_bin_5"][i_com] = fea_com[i_com, 27]
            graph["sp_point_hue_bin_6"][i_com] = fea_com[i_com, 28]
            graph["sp_point_hue_bin_7"][i_com] = fea_com[i_com, 29]
            graph["sp_point_hue_bin_8"][i_com] = fea_com[i_com, 30]
            graph["sp_point_hue_bin_9"][i_com] = fea_com[i_com, 31]
            graph["sp_point_hue_bin_10"][i_com] = fea_com[i_com, 32]
            graph["sp_point_hue_bin_11"][i_com] = fea_com[i_com, 33]
            graph["sp_point_hue_bin_12"][i_com] = fea_com[i_com, 34]
            graph["sp_point_hue_bin_13"][i_com] = fea_com[i_com, 35]
            graph["sp_point_hue_bin_14"][i_com] = fea_com[i_com, 36]

            graph["sp_point_sat_bin_0"][i_com] = fea_com[i_com, 37]
            graph["sp_point_sat_bin_1"][i_com] = fea_com[i_com, 38]
            graph["sp_point_sat_bin_2"][i_com] = fea_com[i_com, 39]
            graph["sp_point_sat_bin_3"][i_com] = fea_com[i_com, 40]
            graph["sp_point_sat_bin_4"][i_com] = fea_com[i_com, 41]

            graph["sp_point_val_bin_0"][i_com] = fea_com[i_com, 42]
            graph["sp_point_val_bin_1"][i_com] = fea_com[i_com, 43]
            graph["sp_point_val_bin_2"][i_com] = fea_com[i_com, 44]
            graph["sp_point_val_bin_3"][i_com] = fea_com[i_com, 45]
            graph["sp_point_val_bin_4"][i_com] = fea_com[i_com, 46]

    # ---compute the superedges features---
    print("    Attaching superedges features ...")
    n_real_edge = 0
    edge_visited_dict = dict()
    for i_sedg in progressbar.progressbar(range(n_sedg), redirect_stdout=True):
    #for i_sedg in range(0, n_sedg):
        i_edg_begin = jump_edg[i_sedg]
        i_edg_end = jump_edg[i_sedg + 1]
        ver_source = edges_added[0, range(i_edg_begin, i_edg_end)]
        ver_target = edges_added[1, range(i_edg_begin, i_edg_end)]
        com_source = edge_comp[0, i_edg_begin]
        com_target = edge_comp[1, i_edg_begin]
        xyz_source = xyz[ver_source, :]
        xyz_target = xyz[ver_target, :]
        if com_source == com_target:
            continue
        if ((com_source, com_target) not in edge_visited_dict) \
                or ((com_target, com_source) not in edge_visited_dict):
            edge_visited_dict[(com_source, com_target)] = True
            edge_visited_dict[(com_target, com_source)] = True
            n_real_edge += 1
            graph["source"][i_sedg] = com_source
            graph["target"][i_sedg] = com_target

            # print(com_source, com_target, len(xyz_source))
            # ---compute the offset set---
            delta = xyz_source - xyz_target
            if len(delta) > 1:
                graph["se_delta_mean"][i_sedg] = np.mean(delta, axis=0)
                graph["se_delta_std"][i_sedg] = np.std(delta, axis=0)
            else:
                graph["se_delta_mean"][i_sedg, :] = delta
                graph["se_delta_std"][i_sedg, :] = [0, 0, 0]
    print("Graph nodes", n_com, ", Graph edges all: ", n_real_edge, ", Segment graph edges: ", n_sedg)
    return graph
