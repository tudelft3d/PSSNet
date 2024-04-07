"""
    PSSNet: Planarity-sensible Semantic Segmentation of Large-scale Urban Meshes
    https://www.sciencedirect.com/science/article/pii/S0924271622003355
    2023 Weixiao GAO, Liangliang Nana, Bas Boom, Hugo Ledoux
    
    Script for converting over-segmentation results and graphs to h5 format
"""
import os.path
import sys
import numpy as np
import argparse
from timeit import default_timer as timer

sys.path.append("./partition/segment_parsing_spg/python_parsing/src")
sys.path.append("./partition/ply_c")
sys.path.append("./partition")

import libpython_parsing
from pssnet_graphs import *
from pssnet_provider import *

parser = argparse.ArgumentParser(description='PSSNet: Planarity-sensible Semantic Segmentation of Large-scale Urban Meshes')
parser.add_argument('--ROOT_PATH', default='../datasets/custom_set_sumv2')
parser.add_argument('--dataset', default='custom_dataset', help='data')
parser.add_argument('--d_se_max', default=0, type=float, help='max length of super edges')
parser.add_argument('--overwrite', default=0, type=int, help='Wether to read existing files or overwrite them')
args = parser.parse_args()

# path to data
root = args.ROOT_PATH + '/'
# list of subfolders to be processed
folders = ["train/", "test/", "validate/"] # "train/", "test/", "validate/","validate_for_test/"
n_labels = 12 # number of classes: h3d: 11, merged: 4, sum: 6, sumv2: 12

times = [0, 0, 0]  # time for computing: features / partition / spg

if not os.path.isdir(root + "clouds"):
    os.mkdir(root + "clouds")
if not os.path.isdir(root + "features"):
    os.mkdir(root + "features")

if not os.path.isdir(root + "superpoint_graphs"):
    os.mkdir(root + "superpoint_graphs")

for folder in folders:
    print("=================\n   " + folder + "\n=================")

    data_folder = root + "data/" + folder
    cloud_folder = root + "clouds/" + folder
    fea_folder = root + "features/" + folder
    spg_folder = root + "superpoint_graphs/" + folder

    pssnetg_folder = root + "pssnet_graphs/" + folder
    if not os.path.isdir(data_folder):
        raise ValueError("%s does not exist" % data_folder)

    if not os.path.isdir(cloud_folder):
        os.mkdir(cloud_folder)
    if not os.path.isdir(fea_folder):
        os.mkdir(fea_folder)
    if not os.path.isdir(spg_folder):
        os.mkdir(spg_folder)

    # list all ply files in the folder
    files = glob.glob(data_folder + "*.ply")
    pssnetg_files = glob.glob(pssnetg_folder + "*.ply")

    if (len(files) == 0):
        raise ValueError('%s is empty' % data_folder)
    if (len(files) == 0):
        raise ValueError('%s is empty' % pssnetg_folder)
    if (len(files) != len(pssnetg_files)):
        raise ValueError('Files in %s and %s are not all match' %(data_folder, pssnetg_folder))

    n_files = len(files)
    i_file = 0

    i = 1
    for file in files:
        i += 1
        file_name = os.path.splitext(os.path.basename(file))[0]
        file_name_splits = file_name.split("_")
        file_name_base = file_name_splits[0] + "_" + file_name_splits[1] + "_" + file_name_splits[2]
        pssnetg_file_name = file_name_base + "_graph"#os.path.splitext(os.path.basename(mfile))[0]

        # adapt to your hierarchy. The following 4 files must be defined
        data_file = data_folder + file_name + '.ply'  # or .las
        cloud_file = cloud_folder + file_name
        fea_file = fea_folder + file_name + '.h5'
        spg_file = spg_folder + file_name + '.h5'
        pssnetg_file = pssnetg_folder + pssnetg_file_name + '.ply'
        i_file = i_file + 1
        print(str(i_file) + " / " + str(n_files) + "---> " + file_name)

        #read data
        xyz, rgb, labels = read_ply(data_file)
        rgb = 255 * rgb  # Now scale by 255
        rgb = rgb.astype(np.uint8)

        start = timer()
        #read over-segmentation and features of points and segments
        components, in_component, geof_mesh, fea_com = libpython_parsing.pointlcoud_parsing(data_file)
        end = timer()
        times[0] = times[0] + end - start
        # --- build the geometric feature file h5 file ---
        # these features will be used for compute average superpoint fecompute_geofature in the learning process
        if os.path.isfile(fea_file) and not args.overwrite:
            print("    reading the existing feature file...")
            geof_mesh, xyz, rgb, labels = read_pssnet_features(fea_file)
        else:
            print("    creating the feature file...")
            write_pssnet_features(fea_file, geof_mesh, xyz, rgb, labels)
        # --compute the partition------
        sys.stdout.flush()
        if os.path.isfile(spg_file) and not args.overwrite:
            print("    reading the existing superpoint graph file...")
            graph_sp, components, in_component = read_pssnet_spg(spg_file)
        else:
            print("    parsing the superpoint graph...")
            # --- build the spg h5 file --
            start = timer()

            components = np.array(components, dtype='object')
            end = timer()

            times[1] = times[1] + end - start
            print("        computation of the SPG...")
            start = timer()

            pssnetg_center, pssnetg_edges = read_graph_ply(pssnetg_file)
            del pssnetg_center

            graph_sp = compute_pssnet_sp_graph(xyz, args.d_se_max, in_component, components, pssnetg_edges, labels, n_labels, fea_com)

            end = timer()
            times[2] = times[2] + end - start
            write_pssnet_spg(spg_file, graph_sp, components, in_component)
        print("Timer : %5.1f / %5.1f / %5.1f " % (times[0], times[1], times[2]))
