import numpy as np
import argparse
import os
parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--name', default='SWA_MRNet', type=str, help='save model path')
opt = parser.parse_args()
result = './snapshots/%s/result.txt'%opt.name

for iter in np.arange(1,11):
    os.system('echo %d | tee -a %s'%(iter*10000, result))
    os.system('python evaluate_cityscapes.py --restore ./snapshots/%s/GTA5_%d_average.pth --batchsize 12 | tee -a %s'%(opt.name,iter*10000, result))
