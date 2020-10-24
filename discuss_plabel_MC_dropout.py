import argparse	
import scipy
from scipy import ndimage
import numpy as np
import sys
import re
from packaging import version

import torch
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data, model_zoo
from model.deeplab import Res_Deeplab
from model.deeplab_multi import DeeplabMulti
from model.deeplab_vgg import DeeplabVGG
from dataset.cityscapes_dataset import cityscapesDataSet
from collections import OrderedDict
import os
from PIL import Image
from utils.tool import fliplr
import matplotlib.pyplot as plt
import torch.nn as nn
import yaml
import copy

torch.backends.cudnn.benchmark=True

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

DATA_DIRECTORY = './data/Cityscapes/data'
DATA_LIST_PATH = './dataset/cityscapes_list/train.txt'
SAVE_PATH = './data/Cityscapes/data/pseudo/train'

if not os.path.isdir('./data/Cityscapes/data/pseudo/'):
    os.mkdir('./data/Cityscapes/data/pseudo/')
    os.mkdir(SAVE_PATH)

IGNORE_LABEL = 255
NUM_CLASSES = 19
NUM_STEPS = 2975 # Number of images in the validation set.
RESTORE_FROM = 'http://vllab.ucmerced.edu/ytsai/CVPR18/GTA2Cityscapes_multi-ed35151c.pth'
RESTORE_FROM_VGG = 'http://vllab.ucmerced.edu/ytsai/CVPR18/GTA2Cityscapes_vgg-ac4ac9f6.pth'
RESTORE_FROM_ORC = 'http://vllab1.ucmerced.edu/~whung/adaptSeg/cityscapes_oracle-b7b9934.pth'
SET = 'train' # We generate pseudo label for training set

MODEL = 'DeeplabMulti'

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

def draw_hist(x, y, xx, yy, name = 'Var'):
    fig = plt.figure()
    bins = np.linspace(0, 1, 50)
    x = x.flatten()
    y = y.flatten()
    xx = np.append(xx,x)
    yy = np.append(yy,y)
    print(len(xx))
    #plt.hist(xx, bins, alpha=0.5, label='Positive')
    plt.hist(yy, bins, alpha=0.5, label='Negative')
    plt.legend(loc='upper right')
    fig.savefig('%s_hist.png' % name)
    plt.clf()
    return xx, yy   

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

def save_heatmap(output_name):
    output, name = output_name
    fig = plt.figure()
    plt.axis('off')
    heatmap = plt.imshow(output, cmap='viridis')
    fig.colorbar(heatmap)
    fig.savefig('%s_heatmap.png' % (name.split('.jpg')[0]))
    return

def activate_drop(m, drop = 0.5):
    classname = m.__class__.__name__
    if classname.find('Drop') != -1:
        m.p = drop
        m.train()

def main():
    """Create the model and start the evaluation process."""
    x0 = []
    y0 = []
    x1 = []
    y1 = []
    count = 0
    right_var = 0
    wrong_var = 0

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
    #args.save += model_name

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
    model.eval()
    model.cuda(gpu0)
    model_drop = copy.deepcopy(model)
    model_drop.apply(activate_drop)
    print(model_drop)

    testloader = data.DataLoader(cityscapesDataSet(args.data_dir, args.data_list, crop_size=(512, 1024), resize_size=(1024, 512), mean=IMG_MEAN, scale=False, mirror=False, set=args.set),
                                    batch_size=batchsize, shuffle=False, pin_memory=True, num_workers=4)

    scale = 1.25
    testloader2 = data.DataLoader(cityscapesDataSet(args.data_dir, args.data_list, crop_size=(round(512*scale), round(1024*scale) ), resize_size=( round(1024*scale), round(512*scale)), mean=IMG_MEAN, scale=False, mirror=False, set=args.set),
                                    batch_size=batchsize, shuffle=False, pin_memory=True, num_workers=4)


    gtloader = data.DataLoader(cityscapesDataSet(args.data_dir, args.data_list, crop_size=(65, 129), resize_size=(129, 65), mean=IMG_MEAN, scale=False, mirror=False, set=args.set),
                                    batch_size=batchsize, shuffle=False, pin_memory=True, num_workers=4)
    if version.parse(torch.__version__) >= version.parse('0.4.0'):
        interp = nn.Upsample(size=(1024, 2048), mode='bilinear', align_corners=True)
        interp2 = nn.Upsample(size=(64, 128), mode='nearest')
    else:
        interp = nn.Upsample(size=(1024, 2048), mode='bilinear')

    sm = torch.nn.Softmax(dim = 1)
    log_sm = torch.nn.LogSoftmax(dim = 1)
    kl_distance = nn.KLDivLoss( reduction = 'none')

    for index, img_data in enumerate(zip(testloader, testloader2, gtloader) ):
        batch, batch2, gt = img_data
        image, _, _, name = batch
        image2, _, _, name2 = batch2
        _, gt_label, _, _ = gt

        inputs = image.cuda()
        inputs2 = image2.cuda()
        print('\r>>>>Extracting feature...%04d/%04d'%(index*batchsize, NUM_STEPS), end='')
        if args.model == 'DeepLab':
            with torch.no_grad():
                output1, output2 = model(inputs)
                output1_drop , output2_drop = model_drop(inputs)
                output_batch = sm(0.5* output1 + output2)
                print(output_batch.shape)

                #heatmap_batch = torch.sum(kl_distance(log_sm(output2_drop), sm(output2)), dim=1)
                heatmap_batch = torch.sum(kl_distance(log_sm(output1_drop), sm(output2_drop)), dim=1)
                heatmap_batch = torch.exp(-heatmap_batch) 
                #output1, output2 = model(fliplr(inputs))
                #output1, output2 = fliplr(output1), fliplr(output2)
                #output_batch += interp(sm(0.5 * output1 + output2))
                #del output1, output2, inputs

                #output1, output2 = model(inputs2)
                #output_batch += interp(sm(0.5* output1 + output2))
                #output1, output2 = model(fliplr(inputs2))
                #output1, output2 = fliplr(output1), fliplr(output2)
                #output_batch += interp(sm(0.5 * output1 + output2))
                #del output1, output2, inputs2
                output_batch = output_batch.cpu().data.numpy()
                heatmap_batch = heatmap_batch.cpu().data.numpy()
        elif args.model == 'DeeplabVGG' or args.model == 'Oracle':
            output_batch = model(Variable(image).cuda())
            output_batch = interp(output_batch).cpu().data.numpy()

        #output_batch = output_batch.transpose(0,2,3,1)
        #output_batch = np.asarray(np.argmax(output_batch, axis=3), dtype=np.uint8)
        output_batch = output_batch.transpose(0,2,3,1)
        score_batch = np.max(output_batch, axis=3)
        output_batch = np.asarray(np.argmax(output_batch, axis=3), dtype=np.uint8)
        #output_batch[score_batch<3.2] = 255  #3.2 = 4*0.8
        for i in range(output_batch.shape[0]):
            output = output_batch[i,:,:]
            name_tmp = name[i].split('/')[-1]
            dir_name = name[i].split('/')[-2]
            save_path = args.save + '/' + dir_name
            print('%s/%s' % (save_path, name_tmp))

            # resize to 64*128
            variance_tmp = heatmap_batch[i,:,:]/np.max(heatmap_batch[i,:,:]) 
            score_tmp = score_batch[i,:,:]/np.max(score_batch[i,:,:]) 
            prediction = output
            ground_truth = gt_label[i,:,:].numpy()
            right_mask = prediction==ground_truth
            ignore_mask = ground_truth==255
            wrong_mask = np.logical_and( (~right_mask), (~ignore_mask))
            # Use High-confidence or not
            #high_mask = score_tmp > 0.95
            #wrong_mask = np.logical_and( wrong_mask, high_mask)
            #right_mask = np.logical_and( right_mask, high_mask)
            right_var += np.mean( variance_tmp[right_mask]) 
            wrong_var += np.mean( variance_tmp[wrong_mask]) 
            count += 1 
            print( right_var/count, wrong_var/count) 
            x0, y0 = draw_hist(variance_tmp[right_mask],variance_tmp[wrong_mask], x0, y0, name='Var')
            #x1, y1 = draw_hist(score_tmp[right_mask],score_tmp[wrong_mask], x1, y1, name='Bias')
    return args.save

if __name__ == '__main__':
    with torch.no_grad():
        save_path = main()
    #os.system('python compute_iou.py ./data/Cityscapes/data/gtFine/train %s'%save_path)
