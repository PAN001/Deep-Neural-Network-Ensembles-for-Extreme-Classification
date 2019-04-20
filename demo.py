# from common import *
from label_category_transform import *
from net.excited_inception_v3 import *
from net.inception_v3 import Inception3
from torch.autograd import Variable
import torch.nn.functional as F
import torch
import torch.nn as nn
import cv2
import os
import numpy as np

# cdiscount data set ----
CDISCOUNT_NUM_CLASSES = 5270
CDISCOUNT_HEIGHT=180
CDISCOUNT_WIDTH =180




## input image preprocessing (mean and std normalisation) ##----------------
def pytorch_image_to_tensor_transform(image):
    mean = [0.485, 0.456, 0.406 ]
    std  = [0.229, 0.224, 0.225 ]
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = image.transpose((2,0,1))
    tensor = torch.from_numpy(image).float().div(255)
    tensor[0] = (tensor[0] - mean[0]) / std[0]
    tensor[1] = (tensor[1] - mean[1]) / std[1]
    tensor[2] = (tensor[2] - mean[2]) / std[2]
    return tensor

def image_to_tensor_transform(image):
    tensor = pytorch_image_to_tensor_transform(image)
    tensor[ 0] = tensor[ 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
    tensor[ 1] = tensor[ 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
    tensor[ 2] = tensor[ 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
    return tensor


# main #################################################################
'''
e.g. you should see this results for : 'LB=0.69565_inc3_00075000_model.pth'


img_file=/media/ssd/data/kaggle/cdiscount/image/train/1000019388/2453177-1.jpg
label =  4572, category_id = 1000006467, prob = 0.211703

img_file=/media/ssd/data/kaggle/cdiscount/image/train/1000012993/6287-0.jpg
label =  1700, category_id = 1000012993, prob = 0.955671

img_file=/media/ssd/data/kaggle/cdiscount/image/train/1000011423/2196367-1.jpg
label =  2560, category_id = 1000011423, prob = 0.951876

img_file=/media/ssd/data/kaggle/cdiscount/image/train/1000005928/5987233-0.jpg
label =   779, category_id = 1000005928, prob = 0.706729


'''


if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    #load a newtork
    #model_file ='/root/share/project/kaggle/cdiscount/deliver/release/trained_models/LB=0.69673_se-inc3_00026000_model.pth'
    #net = SEInception3(in_shape=(3,CDISCOUNT_HEIGHT,CDISCOUNT_WIDTH),num_classes=CDISCOUNT_NUM_CLASSES)

    model_file ='../checkpoint/best_model.pth'
    net = Inception3(in_shape=(3,CDISCOUNT_HEIGHT,CDISCOUNT_WIDTH),num_classes=CDISCOUNT_NUM_CLASSES)

    if os.path.isfile(model_file):
        print("=> loading checkpoint '{}'".format(model_file))
        checkpoint = torch.load(model_file)
        start_epoch = checkpoint['epoch']
        best_train_acc = checkpoint['best_train_acc']
        best_valid_acc = checkpoint['best_valid_acc']
        net.load_state_dict(checkpoint['state_dict']) # load model weights from the checkpoint
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(model_file, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(model_file))
        exit(0)

    net.cuda().eval()


    #let's test a few images
    img_files=[
        'demo_images/1000019388/2453177-1.jpg',
        'demo_images/1000012993/6287-0.jpg',
        'demo_images/1000011423/2196367-1.jpg',
        'demo_images/1000005928/5987233-0.jpg',
    ]

    for img_file in img_files:

        image = cv2.imread(img_file)
        x = image_to_tensor_transform(image)
        x = Variable(x.unsqueeze(0),volatile=True).cuda()

        logits = net(x)
        probs  = F.softmax(logits)

        probs  = probs.cpu().data.numpy().reshape(-1)
        label  = np.argmax(probs)


        print ('img_file=%s'%img_file)
        print ('label = %5d, category_id = %d, prob = %f\n'%(
            label,label_to_category_id[label],probs[label]))



    print('\nsucess!')