import unittest
import numpy as np
import tools as tl
import cv_functions as cvfun
import yaml


class Test(unittest.TestCase):
    """
    The basic class that inherits unittest.TestCase
    """
    algorithm = 'CRep'
    K = 3
    in_folder = '../data/input/'
    out_folder = '../data/output/5-fold_cv/'
    end_file = '_test'
    adj = 'syn111.dat'
    ego = 'source'
    alter = 'target'
    # force_dense = True
    flag_conv = 'log'
    NFold = 5
    out_mask = False
    out_results = True
    out_inference = True

    prng = np.random.RandomState(seed=17)  # set seed random number generator
    rseed = prng.randint(1000)

    ''' 
    Setting to run the algorithm
    '''
    with open('setting_' + algorithm + '.yaml') as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
        conf['out_folder'] = out_folder

    '''
    Import data
    '''
    network = in_folder + adj  # network complete path
    A, B, B_T, data_T_vals = tl.import_data(network, ego=ego, alter=alter, force_dense=True, header=0)
    nodes = A[0].nodes()

    def test_running_algorithm(self):
        print("\nStart running algorithm test\n")

        L = self.B.shape[0]
        N = self.B.shape[1]

        indices = cvfun.shuffle_indices_all_matrix(N, L, rseed=self.rseed)

        for fold in range(self.NFold):
            mask = cvfun.extract_mask_kfold(indices, N, fold=fold, NFold=self.NFold)

            '''
            Set up training dataset    
            '''
            B_train = self.B.copy()
            print(B_train.shape, mask.shape)
            B_train[mask > 0] = 0

            self.conf['end_file'] = '_' + str(fold) + 'K' + str(self.K) + self.end_file
            u, v, w, eta, maxPSL, algo_obj = cvfun.fit_model(B_train, self.B_T, self.data_T_vals, nodes=self.nodes,
                                                             N=N, L=L, K=self.K, algo=self.algorithm,
                                                             flag_conv=self.flag_conv, **self.conf)

            '''
            Load parameters
            '''            
            theta = np.load(self.out_folder+'theta_'+str(fold)+'K'+str(self.K)+self.end_file+'.npz')
            thetaGT = np.load(self.out_folder+'theta_'+str(self.algorithm)+'_'+str(fold)+'K'+str(self.K)+'.npz')

            self.assertTrue(np.array_equal(u, theta['u']))
            self.assertTrue(np.array_equal(v, theta['v']))
            self.assertTrue(np.array_equal(w, theta['w']))
            self.assertTrue(np.array_equal(algo_obj.eta_f, theta['eta']))

            self.assertTrue(np.array_equal(thetaGT['u'], theta['u']))
            self.assertTrue(np.array_equal(thetaGT['v'], theta['v']))
            self.assertTrue(np.array_equal(thetaGT['w'], theta['w']))
            self.assertTrue(np.array_equal(thetaGT['eta'], theta['eta']))


if __name__ == '__main__':
    # begin the unittest.main()
    unittest.main()
