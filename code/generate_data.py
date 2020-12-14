"""
    It generates n synthetic samples of a network having an intrinsic community structure and a given reciprocity value.
    It uses the given yaml setting file.
"""

import generative_model_reciprocity as gm
import yaml
import os
from argparse import ArgumentParser
import numpy as np


def main_generate_data():

    p = ArgumentParser()
    p.add_argument('-s', '--setting', type=str, default='setting_syn_data.yaml')  # file with the setting
    p.add_argument('-n', '--samples', type=int, default=1)  # number of synthetic samples
    args = p.parse_args()

    prng = np.random.RandomState(seed=17)  # set seed random number generator

    with open(args.setting) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)

    out_folder = conf['out_folder']
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    for sn in range(args.samples):
        conf['seed'] += prng.randint(500)
        with open(out_folder + 'setting'+str(conf['seed'])+'.yaml', 'w') as f:
            yaml.dump(conf, f)
        conf['outfile_adj'] = 'syn' + str(conf['seed']) + '.dat'
        gen = gm.GM_reciprocity(**conf)
        _ = gen.reciprocity_planted_network()


if __name__ == '__main__':
    main_generate_data()
