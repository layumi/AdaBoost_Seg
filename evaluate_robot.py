import argparse
import scipy
from scipy import ndimage
import numpy as np
import sys
from packaging import version
from multiprocessing import Pool
import torch
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data, model_zoo
from model.deeplab import Res_Deeplab
from model.deeplab_multi import DeeplabMulti
from model.deeplab_vgg import DeeplabVGG
from dataset.robot_dataset import robotDataSet
from collections import OrderedDict
import os
from PIL import Image
from utils.tool import fliplr
import matplotlib.pyplot as plt
import torch.nn as nn
import yaml
import time

torch.backends.cudnn.benchmark=True

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

DATA_DIRECTORY = './data/Oxford_Robot_ICCV19'
DATA_LIST_PATH = './dataset/robot_list/val.txt'
SAVE_PATH = './result/robot'

IGNORE_LABEL = 255
NUM_CLASSES = 9
NUM_STEPS = 271 # Number of images in the validation set.
RESTORE_FROM = 'http://vllab.ucmerced.edu/ytsai/CVPR18/GTA2Cityscapes_multi-ed35151c.pth'
RESTORE_FROM_VGG = 'http://vllab.ucmerced.edu/ytsai/CVPR18/GTA2Cityscapes_vgg-ac4ac9f6.pth'
RESTORE_FROM_ORC = 'http://vllab1.ucmerced.edu/~whung/adaptSeg/cityscapes_oracle-b7b9934.pth'
SET = 'val'

MODEL = 'DeeplabMulti'

palette = [    
    [70,130,180],
    [220,20,60],
    [119,11,32],
    [0,0,142],
    [220,220,0],
    [250,170,30],
    [70,70,70],
    [244,35,232],
    [128,64,128],
]
palette = [item for sublist in palette for item in sublist]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="Model Choice (DeeplabMulti/DeeplabVGG/Oracle).")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the Cityscapes dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument("--batchsize", type=int, default=12,
                        help="choose gpu device.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose evaluation set.")
    parser.add_argument("--save", type=str, default=SAVE_PATH,
                        help="Path to save result.")
    return parser.parse_args()

def save(output_name):
    output, name = output_name
    output_col = colorize_mask(output)
    output = Image.fromarray(output)

    output.save('%s' % (name))
    output_col.save('%s_color.png' % (name.split('.jpg')[0]))
    return


def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()

    config_path = os.path.join(os.path.dirname(args.restore_from),'opts.yaml')
    with open(config_path, 'r') as stream:
        config = yaml.load(stream)

    args.model = config['model']
    print('ModelType:%s'%args.model)
    print('NormType:%s'%config['norm_style'])
    gpu0 = args.gpu
    batchsize = args.batchsize

    model_name = os.path.basename( os.path.dirname(args.restore_from) )
    args.save += model_name

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    if args.model == 'DeepLab':
        model = DeeplabMulti(num_classes=args.num_classes, use_se = config['use_se'], train_bn = False, norm_style = config['norm_style'])
    elif args.model == 'Oracle':
        model = Res_Deeplab(num_classes=args.num_classes)
        if args.restore_from == RESTORE_FROM:
            args.restore_from = RESTORE_FROM_ORC
    elif args.model == 'DeeplabVGG':
        model = DeeplabVGG(num_classes=args.num_classes)
        if args.restore_from == RESTORE_FROM:
            args.restore_from = RESTORE_FROM_VGG

    if args.restore_from[:4] == 'http' :
        saved_state_dict = model_zoo.load_url(args.restore_from)
    else:
        saved_state_dict = torch.load(args.restore_from)

    try:
        model.load_state_dict(saved_state_dict)
    except:
        model = torch.nn.DataParallel(model)
        model.load_state_dict(saved_state_dict)
    #model = torch.nn.DataParallel(model)
    model.eval()
    model.cuda(gpu0)

    th = 960
    tw = 1280

    testloader = data.DataLoader(robotDataSet(args.data_dir, args.data_list, crop_size=(th, tw), resize_size=(tw, th), mean=IMG_MEAN, scale=False, mirror=False, set=args.set),
                                    batch_size=batchsize, shuffle=False, pin_memory=True, num_workers=4)

    scale = 0.8
    testloader2 = data.DataLoader(robotDataSet(args.data_dir, args.data_list, crop_size=(round(th*scale), round(tw*scale) ), resize_size=( round(tw*scale), round(th*scale)), mean=IMG_MEAN, scale=False, mirror=False, set=args.set),
                                    batch_size=batchsize, shuffle=False, pin_memory=True, num_workers=4)
    scale = 0.9
    testloader3 = data.DataLoader(robotDataSet(args.data_dir, args.data_list, crop_size=(round(th*scale), round(tw*scale) ), resize_size=( round(tw*scale), round(th*scale)), mean=IMG_MEAN, scale=False, mirror=False, set=args.set),
                                    batch_size=batchsize, shuffle=False, pin_memory=True, num_workers=4)


    if version.parse(torch.__version__) >= version.parse('0.4.0'):
        interp = nn.Upsample(size=(960, 1280), mode='bilinear', align_corners=True)
    else:
        interp = nn.Upsample(size=(960, 1280), mode='bilinear')

    sm = torch.nn.Softmax(dim = 1)
    for index, img_data in enumerate(zip(testloader, testloader2, testloader3) ):
        batch, batch2, batch3 = img_data
        image, _, _, name = batch
        image2, _, _, name2 = batch2
        image3, _, _, name3 = batch3

        inputs = image.cuda()
        inputs2 = image2.cuda()
        inputs3 = Variable(image3).cuda()
        print('\r>>>>Extracting feature...%03d/%03d'%(index*batchsize, NUM_STEPS), end='')
        if args.model == 'DeepLab':
            with torch.no_grad():
                output1, output2 = model(inputs)
                output_batch = interp(sm(0.5* output1 + output2))
                #output_batch = interp(sm(output1))
                #output_batch = interp(sm(output2))
                output1, output2 = model(fliplr(inputs))
                output1, output2 = fliplr(output1), fliplr(output2)
                output_batch += interp(sm(0.5 * output1 + output2))
                #output_batch += interp(sm(output1))
                #output_batch += interp(sm(output2))
                del output1, output2, inputs

                output1, output2 = model(inputs2)
                output_batch += interp(sm(0.5* output1 + output2))
                #output_batch += interp(sm(output1))
                #output_batch += interp(sm(output2))
                output1, output2 = model(fliplr(inputs2))
                output1, output2 = fliplr(output1), fliplr(output2)
                output_batch += interp(sm(0.5 * output1 + output2))
                #output_batch += interp(sm(output1))
                #output_batch += interp(sm(output2))
                del output1, output2, inputs2

                #output1, output2 = model(inputs3)
                #output_batch += interp(sm(0.5* output1 + output2))
                #output1, output2 = model(fliplr(inputs3))
                #output1, output2 = fliplr(output1), fliplr(output2)
                #output_batch += interp(sm(0.5 * output1 + output2))
                #del output1, output2, inputs3

                output_batch = output_batch.cpu().data.numpy()
        elif args.model == 'DeeplabVGG' or args.model == 'Oracle':
            output_batch = model(Variable(image).cuda())
            output_batch = interp(output_batch).cpu().data.numpy()

        output_batch = output_batch.transpose(0,2,3,1)
        output_batch = np.asarray(np.argmax(output_batch, axis=3), dtype=np.uint8)
        output_iterator = []
        for i in range(output_batch.shape[0]):
            output_iterator.append(output_batch[i,:,:])
            name_tmp = name[i].split('/')[-1]
            name[i] = '%s/%s' % (args.save, name_tmp)
        with Pool(4) as p:
            p.map(save, zip(output_iterator, name) )
        del output_batch

    
    return args.save

if __name__ == '__main__':
    tt = time.time()
    with torch.no_grad():
        save_path = main()
    print('Time used: {} sec'.format(time.time()-tt))
    devkit_path='dataset/robot_list'
    os.system('python compute_iou.py ./data/Oxford_Robot_ICCV19/anno %s --devkit_dir %s'%(save_path, devkit_path))
