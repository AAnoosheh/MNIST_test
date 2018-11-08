import argparse
import os
import torch

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--name', required=True, type=str, help='name of the experiment: determines where to store models')

        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--data_root', type=str, default='./datasets', help='path to (existing or intended) image folder location')

        self.parser.add_argument('--kernel_width', type=int, default=5, help='kernel size for convolution layers')

        self.parser.add_argument('--gpu_id', type=int, default=0, help='gpu id: e.g. 0 1 2. use -1 for CPU')
        self.parser.add_argument('--n_threads', default=2, type=int, help='# threads for loading data')

        self.initialized = True  # Ignore

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        # save to the disk
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        if not os.path.exists(expr_dir):
            os.makedirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        if not os.path.exists(file_name):
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                args = sorted(vars(self.opt).items())
                for k, v in args:
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')

        return self.opt
