import numpy as np
import argparse
import os
parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--name', default='SWA_MRNet', type=str, help='save model path')
parser.add_argument('--average', action='store_true', help='using average model')
opt = parser.parse_args()
result = './snapshots/%s/result.txt'%opt.name

for iter in np.arange(1,11):
    if opt.average:
        os.system('echo %d+%s | tee -a %s'%(iter*10000, 'average', result))
        os.system('python evaluate_cityscapes.py --restore ./snapshots/%s/GTA5_%d_average.pth --batchsize 6  | tee -a %s'%(opt.name,iter*10000, result))
    else:
        os.system('echo %d | tee -a %s'%(iter*10000, result))
        os.system('python evaluate_cityscapes.py --restore ./snapshots/%s/GTA5_%d.pth --batchsize 6  | tee -a %s'%(opt.name,iter*10000, result))
