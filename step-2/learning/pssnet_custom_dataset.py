"""
    PSSNet: Planarity-sensible Semantic Segmentation of Large-scale Urban Meshes
    https://www.sciencedirect.com/science/article/pii/S0924271622003355
    2023 Weixiao GAO, Liangliang Nana, Bas Boom, Hugo Ledoux
"""
from __future__ import division
from __future__ import print_function
from builtins import range

import sys

sys.path.append("./learning")

import random
import numpy as np
import os
import functools
import torch
import torchnet as tnt
import h5py
import pssnet_spg  # import spg

from sklearn.linear_model import RANSACRegressor

def get_datasets(args, test_seed_offset=0):
    """build training and testing set"""

    # for a simple train/test organization
    validset = ['validate/' + f for f in os.listdir(args.CUSTOM_SET_PATH + '/superpoint_graphs/validate')]
    trainset = ['train/' + f for f in os.listdir(args.CUSTOM_SET_PATH + '/superpoint_graphs/train')]
    testset = ['test/' + f for f in os.listdir(args.CUSTOM_SET_PATH + '/superpoint_graphs/test')]


    # Load superpoints graphs
    testlist, trainlist, validlist = [], [], []
    for n in trainset:
        trainlist.append(pssnet_spg.pssnet_spg_reader(args, args.CUSTOM_SET_PATH + '/superpoint_graphs/' + n, True))
    for n in validset:
        validlist.append(pssnet_spg.pssnet_spg_reader(args, args.CUSTOM_SET_PATH + '/superpoint_graphs/' + n, True))
    for n in testset:
        testlist.append(pssnet_spg.pssnet_spg_reader(args, args.CUSTOM_SET_PATH + '/superpoint_graphs/' + n, True))

    # Normalize edge features
    if args.spg_attribs01:
        trainlist, testlist, validlist, scaler = pssnet_spg.scaler01(trainlist, testlist, validlist=validlist)

    return tnt.dataset.ListDataset([pssnet_spg.spg_to_igraph(*tlist) for tlist in trainlist],
                                    functools.partial(pssnet_spg.loader, train=True, args=args, db_path=args.CUSTOM_SET_PATH)), \
           tnt.dataset.ListDataset([pssnet_spg.spg_to_igraph(*tlist) for tlist in testlist],
                                    functools.partial(pssnet_spg.loader, train=False, args=args, db_path=args.CUSTOM_SET_PATH, test_seed_offset=test_seed_offset)), \
           tnt.dataset.ListDataset([pssnet_spg.spg_to_igraph(*tlist) for tlist in validlist],
                                    functools.partial(pssnet_spg.loader, train=False, args=args, db_path=args.CUSTOM_SET_PATH, test_seed_offset=test_seed_offset)), \
            scaler

def get_info(args):
    edge_feats = 0
    for attrib in args.edge_attribs.split(','):
        a = attrib.split('/')[0]
        if a in ['delta_avg', 'delta_std', 'xyz']:
            edge_feats += 3
        else:
            edge_feats += 1

    if args.loss_weights == 'none':
        weights = np.ones((12,), dtype='f4')
        #weights = np.ones((6,), dtype='f4')
    if args.loss_weights == 'proportional':
        weights = h5py.File(args.CUSTOM_SET_PATH + "/parsed/class_count.h5")["class_count"][:].astype('f4')
        weights = weights[:, [i for i in range(6) if i != args.cvfold - 1]].sum(1)
        weights = weights.mean() / weights
    if args.loss_weights == 'sqrt':
        weights = np.sqrt(weights)
    if args.loss_weights == 'imbalanced':
        # pre-calculate the number of points in each category
        num_per_class = []
        # np.array([1, 1, 1, 1, 1, 1], dtype=np.int32)
        #segment num: 20650, 1824, 117812, 1017, 12765, 3331
        #H3D: 2584, 1816, 174, 1624, 1886, 1507, 2860, 1113, 400, 563, 171
        #SUMV2: 589720,425717,1061496,310500,38179,25023,485347,50290,19235,25415,26237,15031
        #SUMV2_ segment: 17902,65896,39279,1671,8706,3287,15702,14933,3062,6711,5215,1475
        num_per_class = np.array([5387,1738,15135,331,2230,447,6308,3109,1260,2161,2385,1079], dtype=np.int32)
        weight = num_per_class / float(sum(num_per_class))
        ce_label_weight = 1.0 / weight # weight, 1 / (weight + 0.02)
        #weights = np.expand_dims(ce_label_weight, axis=0).astype(np.float32)
        weights = ce_label_weight.astype(np.float32)
        # add sqrt
        weights = np.sqrt(weights)
    weights = torch.from_numpy(weights).cuda() if args.cuda else torch.from_numpy(weights)

    return {
        'node_feats': args.ptn_nfeat_stn,
        'edge_feats': edge_feats,
        'class_weights': weights,
        'classes': 12,  # CHANGE TO YOUR NUMBER OF CLASS, 6, 12
        #'inv_class_map': {0: 'ground', 1: 'vegetation', 2: 'building', 3: 'water', 4: 'car', 5: 'boat'},  # etc...
        #'inv_class_map': {0: 'ground', 1: 'vegetation', 2: 'building', 3: 'vehicle'},  # C5...
        'inv_class_map': {0: 'terrain', 1: 'high_vegetation', 2: 'facade_surface', 3: 'water',  4: 'car', 5: 'boat', 6: 'roof_surface', 7: 'chimney', 8: 'dormer', 9: 'balcony', 10: 'roof_installation', 11: 'wall'},  # C11...
    }

def preprocess_pointclouds(CUSTOM_SET_PATH):
    """ Preprocesses data by splitting them by components and normalizing."""

    for n in ['train', 'validate', 'test']: #'validate_for_test'   'train', 'validate', 'test'
        pathP = '{}/parsed/{}/'.format(CUSTOM_SET_PATH, n)
        pathD = '{}/features/{}/'.format(CUSTOM_SET_PATH, n)
        pathC = '{}/superpoint_graphs/{}/'.format(CUSTOM_SET_PATH, n)
        if not os.path.exists(pathP):
            os.makedirs(pathP)
        random.seed(0)

        for file in os.listdir(pathC):
            print(file)
            if file.endswith(".h5"):
                f = h5py.File(pathD + file, 'r')
                xyz = f['xyz'][:]
                rgb = f['rgb'][:].astype(np.float32)
                P = np.zeros(1)
                geof = f['geof'][:]  #  point-based feas
                rgb = rgb / 255.0  # - 0.5
                P = np.concatenate([xyz, rgb, geof], axis=1)

                f = h5py.File(pathC + file, 'r')
                numc = len(f['components'].keys())

                with h5py.File(pathP + file, 'w') as hf:
                    for c in range(numc):
                        idx = f['components/{:d}'.format(c)][:].flatten()
                        if idx.size > 10000:  # trim extra large segments, just for speed-up of loading time
                            ii = random.sample(range(idx.size), k=10000)
                            idx = idx[ii]

                        hf.create_dataset(name='{:d}'.format(c), data=P[idx, ...])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs')
    parser.add_argument('--CUSTOM_SET_PATH', default='../datasets/custom_set_sumv2')
    args = parser.parse_args()
    preprocess_pointclouds(args.CUSTOM_SET_PATH)


