import unittest
import numpy as np
import CRep as CREP
import yaml
import tools as tl


class Test(unittest.TestCase):
    """
    The basic class that inherits unittest.TestCase
    """
    algorithm = 'CRep'
    K = 3
    in_folder = '../data/input/'
    out_folder = '../data/output/'
    end_file = '_test'
    adj = 'syn111.dat'
    ego = 'source'
    alter = 'target'
    force_dense = False

    '''
    Import data
    '''
    network = in_folder + adj  # network complete path
    A, B, B_T, data_T_vals = tl.import_data(network, ego=ego, alter=alter, force_dense=force_dense, header=0)
    nodes = A[0].nodes()

    ''' 
    Setting to run the algorithm
    '''
    with open('setting_' + algorithm + '.yaml') as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
        conf['end_file'] = end_file

    model = CREP.CRep(N=A[0].number_of_nodes(), L=len(A), K=K, **conf)

    # test case function to check the crep.set_name function
    def test_import_data(self):
        print("Start import data test\n")
        if self.force_dense:
            self.assertTrue(self.B.sum() > 0)
            print('B has ', self.B.sum(), ' total weight.')
        else:
            self.assertTrue(self.B.vals.sum() > 0)
            print('B has ', self.B.vals.sum(), ' total weight.')

    # test case function to check the Person.get_name function
    def test_running_algorithm(self):
        print("\nStart running algorithm test\n")

        _ = self.model.fit(data=self.B, data_T=self.B_T, data_T_vals=self.data_T_vals, nodes=self.nodes)

        theta = np.load(self.model.out_folder+'theta'+self.model.end_file+'.npz')
        thetaGT = np.load(self.model.out_folder+'theta_'+self.algorithm+'.npz')

        self.assertTrue(np.array_equal(self.model.u_f, theta['u']))
        self.assertTrue(np.array_equal(self.model.v_f, theta['v']))
        self.assertTrue(np.array_equal(self.model.w_f, theta['w']))
        self.assertTrue(np.array_equal(self.model.eta_f, theta['eta']))

        self.assertTrue(np.array_equal(thetaGT['u'], theta['u']))
        self.assertTrue(np.array_equal(thetaGT['v'], theta['v']))
        self.assertTrue(np.array_equal(thetaGT['w'], theta['w']))
        self.assertTrue(np.array_equal(thetaGT['eta'], theta['eta']))


if __name__ == '__main__':
    # begin the unittest.main()
    unittest.main()
