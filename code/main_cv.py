"""
    Main function to implement cross-validation given a number of communities.

    - Hold-out part of the dataset (pairs of edges labeled by unordered pairs (i,j));
    - Infer parameters on the training set;
    - Calculate performance measures in the test set (AUC).
"""

# TODO: optimize for big matrices (so when the input would be done with force_dense=False)

import csv
import os
import pickle
from argparse import ArgumentParser
import cv_functions as cvfun
import numpy as np
import tools as tl
import yaml
import sktensor as skt
import time


def main():
    p = ArgumentParser()
    p.add_argument('-a', '--algorithm', type=str, choices=['Crep', 'Crepnc', 'Crep0'], default='CRep')  # configuration
    p.add_argument('-K', '--K', type=int, default=3)  # number of communities
    p.add_argument('-A', '--adj', type=str, default='syn111.dat')  # name of the network
    p.add_argument('-f', '--in_folder', type=str, default='../data/input/')  # path of the input network
    p.add_argument('-o', '--out_folder', type=str, default='../data/output/5-fold_cv/')  # path to store outputs
    p.add_argument('-e', '--ego', type=str, default='source')  # name of the source of the edge
    p.add_argument('-t', '--alter', type=str, default='target')  # name of the target of the edge
    # p.add_argument('-d', '--force_dense', type=bool, default=True)  # flag to force a dense transformation in input
    p.add_argument('-F', '--flag_conv', type=str, choices=['log', 'deltas'], default='log')  # flag for convergence
    p.add_argument('-N', '--NFold', type=int, default=5)  # number of fold to perform cross-validation
    p.add_argument('-m', '--out_mask', type=bool, default=False)  # flag to output the masks
    p.add_argument('-r', '--out_results', type=bool, default=True)  # flag to output the results in a csv file
    p.add_argument('-i', '--out_inference', type=bool, default=True)  # flag to output the inferred parameters
    args = p.parse_args()

    prng = np.random.RandomState(seed=17)  # set seed random number generator

    '''
    Cross validation parameters and set up output directory
    '''
    NFold = args.NFold
    out_mask = args.out_mask
    out_results = args.out_results

    out_folder = args.out_folder
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    '''
    Model parameters
    '''
    K = args.K
    network = args.in_folder + args.adj  # network complete path
    algorithm = args.algorithm  # algorithm to use to generate the samples
    adjacency = args.adj.split('.dat')[0]  # name of the network without extension
    with open('setting_' + algorithm + '.yaml') as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    conf['out_folder'] = out_folder
    conf['out_inference'] = args.out_inference

    '''
    Import data
    '''
    A, B, B_T, data_T_vals = tl.import_data(network, ego=args.ego, alter=args.alter, force_dense=True, header=0)
    nodes = A[0].nodes()
    valid_types = [np.ndarray, skt.dtensor, skt.sptensor]
    assert any(isinstance(B, vt) for vt in valid_types)

    print('\n### CV procedure ###')
    comparison = [0 for _ in range(11)]
    comparison[0] = K

    # save the results
    if out_results:
        out_file = out_folder + adjacency + '_cv.csv'
        if not os.path.isfile(out_file):  # write header
            with open(out_file, 'w') as outfile:
                wrtr = csv.writer(outfile, delimiter=',', quotechar='"')
                wrtr.writerow(['K', 'fold', 'rseed', 'eta', 'auc_train', 'auc_test', 'auc_cond_train', 'auc_cond_test',
                               'opt_func_train', 'opt_func_test', 'max_it'])
        outfile = open(out_file, 'a')
        wrtr = csv.writer(outfile, delimiter=',', quotechar='"')
        print(f'Results will be saved in: {out_file}')

    time_start = time.time()
    L = B.shape[0]
    N = B.shape[-1]

    rseed = prng.randint(1000)
    indices = cvfun.shuffle_indices_all_matrix(N, L, rseed=rseed)
    init_end_file = conf['end_file']

    for fold in range(NFold):
        print('\nFOLD ', fold)
        comparison[1], comparison[2] = fold, rseed

        mask = cvfun.extract_mask_kfold(indices, N, fold=fold, NFold=NFold)
        if out_mask:
            outmask = out_folder + 'mask_f' + str(fold) + '_' + adjacency + '.pkl'
            print(f'Mask saved in: {outmask}')
            with open(outmask, 'wb') as f:
                pickle.dump(np.where(mask > 0), f)

        '''
        Set up training dataset    
        '''
        B_train = B.copy()
        B_train[mask > 0] = 0

        '''
        Run CRep on the training 
        '''
        tic = time.time()
        conf['end_file'] = init_end_file + '_' + str(fold) + 'K' + str(K)
        u, v, w, eta, maxPSL, algo_obj = cvfun.fit_model(B_train, B_T, data_T_vals, nodes=nodes, N=N, L=L, K=K,
                                                         algo=algorithm, flag_conv=args.flag_conv, **conf)

        '''
        Output performance results
        '''
        comparison[3] = eta
        M = cvfun.calculate_expectation(u, v, w, eta=eta)
        comparison[4] = cvfun.calculate_AUC(M, B, mask=np.logical_not(mask))
        comparison[5] = cvfun.calculate_AUC(M, B, mask=mask)
        M_cond = cvfun.calculate_conditional_expectation(B, u, v, w, eta=eta)
        comparison[6] = cvfun.calculate_AUC(M_cond, B, mask=np.logical_not(mask))
        comparison[7] = cvfun.calculate_AUC(M_cond, B, mask=mask)
        comparison[9] = cvfun.calculate_opt_func(B, algo_obj, mask=mask, assortative=conf['assortative'])
        comparison[8] = maxPSL
        comparison[10] = algo_obj.final_it

        print(f'Time elapsed: {np.round(time.time() - tic, 2)} seconds.')

        if out_results:
            wrtr.writerow(comparison)
            outfile.flush()

    if out_results:
        outfile.close()

    print(f'\nTime elapsed: {np.round(time.time() - time_start, 2)} seconds.')


if __name__ == '__main__':
    main()
