from __future__ import print_function

import os
from torch.autograd import Variable
import torch.nn.functional as F
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from transform import *
from Utils import *
from cdimage import *
from torch.utils.data.sampler import RandomSampler
import operator
from tqdm import tqdm
import label_category_transform
from get_net import *
# --------------------------------------------------------
from net.resnet101 import ResNet101 as ResNet
from net.xception import Xception as XcepNet
from net.inception_v3 import Inception3 as IncNet

def wrap(image):
    return ResNet.valid_augment(random_shift_scale_rotate(image))

TTA_list = [ResNet.valid_augment, IncNet.valid_augment, XcepNet.valid_augment]
# TTA_list = [fix_center_crop, random_shift_scale_rotate]
# TTA_list = [ResNet.valid_augment, wrap]
# TTA_list = [IncNet.valid_augment]
# TTA_list = [XcepNet.valid_augment]
transform_num = len(TTA_list)
TTA_threshold = 0.5

use_cuda = True
IDENTIFIER = "resnet"
SEED = 123456
PROJECT_PATH = './project'
CDISCOUNT_HEIGHT = 180
CDISCOUNT_WIDTH = 180
CDISCOUNT_NUM_CLASSES = 5270

csv_dir = './data/'
root_dir = '../output/'
test_data_filename = 'test.csv'
validation_small_data_filename = 'validation_small.csv'
validation_large_data_filename = 'validation.csv'

resnet_initial_checkpoint = "./latest/resnet/latest.pth"
inc3_initial_checkpoint = "./latest/inc3/latest.pth"
xce3_initial_checkpoint = "./latest/xception_correct_norm/latest.pth"
resnet_pseudo_initial_checkpoint = "./latest/resnet_pseudo/latest.pth"

net_params = {"in_shape": (3, CDISCOUNT_HEIGHT, CDISCOUNT_WIDTH), "num_classes": CDISCOUNT_NUM_CLASSES}

res_path = "./test_res/" + IDENTIFIER + "_test_TTA_ensemble_all.res"
validation_batch_size = 56

def ensemble_predict(cur_procuct_probs, num):
    candidates = list(set(np.argmax(cur_procuct_probs, axis=1))) # remove dups
    if len(candidates) == 1:
        return candidates[0]

    print("candidates: ", candidates)
    probs_means = np.mean(cur_procuct_probs, axis=0)
    winner_score = 0.0
    winner = None
    for candidate in candidates:
        # Adopt criteria to abandan some instances
        print("=> candidate: ", candidate)
        print("=> prob_mean: ", probs_means[candidate])
        candidate_score = probs_means[candidate] * num
        abandan_cnt = 0
        for probs in cur_procuct_probs:  # iterate each product instance
            print("prob: ", probs[candidate])
            other_mean = (probs_means[candidate] * num - probs[candidate]) / (num - 1)
            if probs[candidate] < other_mean * TTA_threshold or probs[candidate] > other_mean * (1 / TTA_threshold):
                # abandan this instance
                candidate_score -= probs[candidate]
                abandan_cnt += 1

        if abandan_cnt == num:
            candidate_score = probs_means[candidate]
        else:
            candidate_score = float(candidate_score) / (num - abandan_cnt)

        if candidate_score > winner_score:
            winner = candidate
            winner_score = candidate_score

    return winner

def TTA(images):
    images_TTA_list = []

    for transform in TTA_list:
        cur_images = []
        for image in images:
            cur_images.append(transform(image))

        images_TTA_list.append(torch.stack(cur_images))

    return images_TTA_list

# def TTA_separate(images):
#     images_TTA_list = []
#
#     for transform in TTA_list:
#         cur_images = []
#         for image in images:
#             cur_images.append(pytorch_image_to_tensor_transform(transform(image)))
#
#         images_TTA_list.append(torch.stack(cur_images))
#
#     return images_TTA_list


def evaluate_sequential_ensemble_val(net, loader, path):
    cur_procuct_probs = []
    cur_product_id = None
    cur_product_label = None

    correct_product_cnt = 0
    total_product_cnt = 0

    for iter, (images, labels, image_ids) in enumerate(tqdm(loader), 0):
        # if total_product_cnt > 10:
        #     break

        labels = labels.numpy()
        image_ids = np.array(image_ids)

        # transforms
        images_list = TTA(images.numpy()) # a list of image batch using different transforms
        probs_list = []
        for images in images_list:
            images = Variable(images.type(torch.FloatTensor)).cuda()
            logits = net(images)
            probs  = (((F.softmax(logits)).cpu().data.numpy()).astype(float))
            probs_list.append(probs)

        i = 0
        for image_id in image_ids:
            product_id = imageid_to_productid(image_id)

            if cur_product_id == None:
                cur_product_id = product_id
                cur_product_label = labels[i]

            if product_id != cur_product_id:
                # a new product
                print("------------------------- cur product: " + str(cur_product_id) + "-------------------------")

                # find winner for previous product
                num = len(cur_procuct_probs) * transform_num # total number of instances for current product
                print("Number of instances: ", num)

                # do predictions
                cur_procuct_probs = np.array(cur_procuct_probs)
                winner = ensemble_predict(cur_procuct_probs, num)

                if winner == cur_product_label:
                    correct_product_cnt += 1
                print("winner: ", str(winner))
                print("label: ", str(cur_product_label))

                total_product_cnt += 1

                print("Acc: ", str(float(correct_product_cnt) / total_product_cnt))

                # update
                cur_product_id = product_id
                cur_product_label = labels[i]
                cur_procuct_probs = []

            for probs in probs_list:
                cur_procuct_probs.append(probs[i])

            i += 1

    # find winner for current product
    num = len(cur_procuct_probs) * transform_num  # total number of instances for current product
    # do predictions
    winner = ensemble_predict(np.array(cur_procuct_probs), num)

    if winner == cur_product_label:
        correct_product_cnt += 1

    total_product_cnt += 1

    print("Acc: ", str(float(correct_product_cnt) / total_product_cnt))


def evaluate_sequential_ensemble_val_bagging(nets, loader, path):
    cur_procuct_probs = []
    cur_product_id = None
    cur_product_label = None

    correct_product_cnt = 0
    total_product_cnt = 0

    for iter, (images, labels, image_ids) in enumerate(tqdm(loader), 0):
        # if total_product_cnt > 10:
        #     break

        labels = labels.numpy()
        image_ids = np.array(image_ids)

        # transforms
        images_list = TTA(images.numpy()) # a list of image batch using different transforms
        assert (len(images_list) == transform_num)
        probs_list = []

        i = 0
        for images in images_list:
            images = Variable(images.type(torch.FloatTensor)).cuda()
            net = nets[0]
            logits = net(images)
            probs  = (((F.softmax(logits)).cpu().data.numpy()).astype(float))
            probs_list.append(probs)
            i += 1

        i = 0
        for image_id in image_ids:
            product_id = imageid_to_productid(image_id)

            if cur_product_id == None:
                cur_product_id = product_id
                cur_product_label = labels[i]

            if product_id != cur_product_id:
                # a new product
                print("------------------------- cur product: " + str(cur_product_id) + "-------------------------")

                # find winner for previous product
                num = len(cur_procuct_probs) * transform_num # total number of instances for current product
                print("Number of instances: ", num)

                # do predictions
                cur_procuct_probs = np.array(cur_procuct_probs)
                winner = ensemble_predict(cur_procuct_probs, num)

                if winner == cur_product_label:
                    correct_product_cnt += 1
                print("winner: ", str(winner))
                print("label: ", str(cur_product_label))

                total_product_cnt += 1

                print("Acc: ", str(float(correct_product_cnt) / total_product_cnt))

                # update
                cur_product_id = product_id
                cur_product_label = labels[i]
                cur_procuct_probs = []

            for probs in probs_list:
                cur_procuct_probs.append(probs[i])

            i += 1

    # find winner for current product
    num = len(cur_procuct_probs) * transform_num  # total number of instances for current product
    # do predictions
    winner = ensemble_predict(np.array(cur_procuct_probs), num)

    if winner == cur_product_label:
        correct_product_cnt += 1

    total_product_cnt += 1

    print("Acc: ", str(float(correct_product_cnt) / total_product_cnt))


def load_net(identifier, initial_checkpoint, net_params):
    net = get_net(identifier, net_params)
    net.cuda()
    net.eval()

    if os.path.isfile(initial_checkpoint):
        print("=> loading checkpoint '{}'".format(initial_checkpoint))

        # load checkpoint
        checkpoint = torch.load(initial_checkpoint)
        net.load_state_dict(checkpoint['state_dict'])  # load model weights from the checkpoint

        # # load pretrained model
        # net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))

        print("=> loaded checkpoint '{}'".format(initial_checkpoint))
    else:
        print("=> no checkpoint found at '{}'".format(initial_checkpoint))
        exit(0)

    return net

def evaluate_sequential_ensemble_test(nets, loader, path):
    product_to_prediction_map = {}
    cur_procuct_probs = []
    cur_product_id = None

    with open(path, "a") as file:
        file.write("_id,category_id\n")

        for iter, (images, _, image_ids) in enumerate(tqdm(loader), 0):
            image_ids = np.array(image_ids)

            # transforms
            images_list = TTA(images.numpy()) # a list of image batch using different transforms
            probs_list = []
            i = 0
            for images in images_list:
                images = Variable(images.type(torch.FloatTensor)).cuda()
                logits = nets[0](images)
                probs  = ((F.softmax(logits)).cpu().data.numpy()).astype(float)
                probs_list.append(probs)
                i += 1

            i = 0
            for image_id in image_ids:
                product_id = imageid_to_productid(image_id)

                if cur_product_id == None:
                    cur_product_id = product_id

                if product_id != cur_product_id:
                    # a new product
                    print("------------------------- cur product: " + str(cur_product_id) + "-------------------------")

                    # find winner for previous product
                    num = len(cur_procuct_probs) * transform_num  # total number of instances for current product
                    print("Number of instances: ", num)

                    # do predictions
                    cur_procuct_probs = np.array(cur_procuct_probs)
                    winner = ensemble_predict(cur_procuct_probs, num)

                    # save winner
                    product_to_prediction_map[cur_product_id] = winner

                    # update
                    cur_product_id = product_id
                    cur_procuct_probs = []

                for probs in probs_list:
                    cur_procuct_probs.append(probs[i])

                i += 1

        # a new product
        print("------------------------- cur product: " + str(cur_product_id) + "-------------------------")

        # find winner for previous product
        num = len(cur_procuct_probs) * transform_num  # total number of instances for current product
        print("Number of instances: ", num)

        # do predictions
        cur_procuct_probs = np.array(cur_procuct_probs)
        winner = ensemble_predict(cur_procuct_probs, num)

        # save winner
        product_to_prediction_map[cur_product_id] = winner

        for product_id, prediction in product_to_prediction_map.items():
            file.write(str(product_id) + "," + str(label_to_category_id[prediction]) + "\n")

# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    res_net = load_net("resnet101", resnet_initial_checkpoint, net_params)
    # res_pseudo_net = load_net("resnet101", resnet_pseudo_initial_checkpoint, net_params)
    inc3_net = load_net("inceptionv3", inc3_initial_checkpoint, net_params)
    xce3_net = load_net("xceptionv3", xce3_initial_checkpoint, net_params)


    # dataset = CDiscountDataset(csv_dir + validation_large_data_filename, root_dir, mode="valid", transform=None)
    dataset = CDiscountDataset(csv_dir + test_data_filename, root_dir, mode="test", transform=None)
    loader  = DataLoader(
                        dataset,
                        sampler=SequentialSampler(dataset),
                        batch_size  = validation_batch_size,
                        drop_last   = False,
                        num_workers = 4,
                        pin_memory  = False)

    nets = [res_net, inc3_net, xce3_net]
    # nets = [res_pseudo_net]
    # nets = [res_net]
    # nets = [inc3_net]
    # nets = [xce3_net]
    product_to_prediction_map = evaluate_sequential_ensemble_test(nets, loader, res_path)

    print('\nsucess!')
