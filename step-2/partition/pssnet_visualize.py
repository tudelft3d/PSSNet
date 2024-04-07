"""
    PSSNet: Planarity-sensible Semantic Segmentation of Large-scale Urban Meshes
    https://www.sciencedirect.com/science/article/pii/S0924271622003355
    2023 Weixiao GAO, Liangliang Nana, Bas Boom, Hugo Ledoux
    
    functions for writing and reading features and graph
"""
import os.path
import numpy as np
import argparse
import sys

sys.path.append("./partition/")
from plyfile import PlyData, PlyElement
from pssnet_provider import *

#from provider import *

parser = argparse.ArgumentParser(description='Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs')
parser.add_argument('--dataset', default='custom_dataset', help='dataset name: sema3d|s3dis')
parser.add_argument('--ROOT_PATH', default='../datasets/custom_set_sumv2', help='folder containing the ./data folder')
parser.add_argument('--res_file', default='../datasets/custom_set_sumv2/results/predictions_test',
                    help='folder containing the results')
parser.add_argument('--upsample', default=1, type=int,
                    help='if 1, upsample the prediction to the original cloud (if the files is huge it can take a very long and use a lot of memory - avoid on sema3d)')
parser.add_argument('--ver_batch', default=0, type=int, help='Batch size for reading large files')

parser.add_argument('--output_type', default='r', help='which cloud to output: i = input rgb pointcloud \
                    , g = ground truth, f = geometric features, p = partition, r = prediction result \
                    , e = error, s = SPG')
args = parser.parse_args()
# ---path to data---------------------------------------------------------------
# root of the data directory
root = args.ROOT_PATH + '/'
rgb_out = 'i' in args.output_type
gt_out = 'g' in args.output_type
fea_out = 'f' in args.output_type
par_out = 'p' in args.output_type
res_out = 'r' in args.output_type
err_out = 'e' in args.output_type
spg_out = 's' in args.output_type
folder = 'test/'

n_labels = 12 #11 #13 #6 #5
plyfiles = glob.glob(root + "data/" + folder + "*.ply")
ply_folder = root + "clouds/" + folder
n_files = len(plyfiles)
i_file = 0
for ply_f in plyfiles:
    ply_f_name = os.path.splitext(os.path.basename(ply_f))[0]
    # adapt to your hierarchy. The following 4 files must be defined
    fea_file = root + "features/" + folder + ply_f_name + '.h5'
    spg_file = root + "superpoint_graphs/" + folder + ply_f_name + '.h5'
    ply_file = ply_folder + ply_f_name
    res_file = args.res_file + '.h5'
    i_file = i_file + 1
    print(str(i_file) + " / " + str(n_files) + "---> " + ply_f_name)

    if not os.path.isdir(root + "clouds/"):
        os.mkdir(root + "clouds/")
    if not os.path.isdir(ply_folder):
        os.mkdir(ply_folder)
    if (not os.path.isfile(fea_file)):
        raise ValueError("%s does not exist and is needed" % fea_file)

    geof, xyz, rgb, labels = read_pssnet_features(fea_file)

    if (par_out or res_out) and (not os.path.isfile(spg_file)):
        raise ValueError("%s does not exist and is needed to output the partition  or result ply" % spg_file)
    else:
        graph_spg, components, in_component = read_pssnet_spg(spg_file)
    if res_out or err_out:
        if not os.path.isfile(res_file):
            raise ValueError("%s does not exist and is needed to output the result ply" % res_file)
        try:
            pred_red = np.array(h5py.File(res_file, 'r').get(folder + ply_f_name))
            if (len(pred_red) != len(components)):
                raise ValueError("It looks like the spg is not adapted to the result file")
            pred_full = reduced_labels2full(pred_red, components, len(xyz))
        except OSError:
            raise ValueError("%s does not exist in %s" % (folder + ply_f_name, res_file))
    # ---write the output clouds----------------------------------------------------
    if rgb_out:
        print("writing the RGB file...")
        write_ply(ply_file + "_rgb.ply", xyz, rgb)

    if gt_out:
        print("writing the GT file...")
        prediction2ply(ply_file + "_GT.ply", xyz, labels, n_labels, args.dataset)

    if fea_out:
        print("writing the features file...")
        geof2ply(ply_file + "_geof.ply", xyz, geof)

    if par_out:
        print("writing the partition file...")
        partition2ply_with_segids(ply_file + "_partition.ply", xyz, components, in_component)

    if res_out and not bool(args.upsample):
        print("writing the prediction file...")
        prediction2ply(ply_file + "_pred.ply", xyz, pred_full + 1, n_labels, args.dataset)

    if err_out:
        print("writing the error file...")
        error2ply(ply_file + "_err.ply", xyz, rgb, labels, pred_full + 1)

    if spg_out:
        print("writing the SPG file...")
        spg2ply(ply_file + "_spg.ply", graph_spg)

    if res_out and bool(args.upsample):
        data_folder = root + 'data/test/'
        data_file = data_folder + ply_f_name + ".ply"
        xyz_up, rgb_up, labels_truth = read_ply(data_file)
        pred_up = interpolate_labels(xyz_up, xyz, pred_full, args.ver_batch)
        print("writing the upsampled prediction file...")
        prediction2ply_withlabels(ply_file + "_pred_up.ply", xyz_up, labels_truth, pred_up + 1, n_labels, args.dataset)

