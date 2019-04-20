import numpy as np
from label_category_transform import *
from transform import *
from torch.autograd import Variable
import torch.nn.functional as F


def general_image_to_tensor_transform(image):
    tensor = pytorch_image_to_tensor_transform(image)
    tensor[ 0] = tensor[ 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
    tensor[ 1] = tensor[ 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
    tensor[ 2] = tensor[ 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
    return tensor

def general_train_augment(image):

    if random.random() < 0.5:
        image = random_shift_scale_rotate(image,
                #shift_limit  = [0, 0],
                shift_limit  = [-0.06,  0.06],
                scale_limit  = [0.9, 1.2],
                rotate_limit = [-10,10],
                aspect_limit = [1,1],
                #size=[1,299],
        borderMode=cv2.BORDER_REFLECT_101 , u=1)
    else:
        pass

    # flip  random ---------
    image = random_horizontal_flip(image, u=0.5)
    #print("enter image_to_tensor_transform")
    tensor = general_image_to_tensor_transform(image)
    return tensor


def general_valid_augment(image):
    tensor = general_image_to_tensor_transform(image)
    return tensor


def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
       lr +=[ param_group['lr'] ]
    return lr

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def imageid_to_productid(image_id):
    splitted = image_id.split("-")
    product_id = splitted[0]

    return product_id

def product_predict_average_prob(image_ids, probses):
    """

    :param probs: A dictionary: {img_id -> [probability_distribution]} where probability_distribution is an array
    :type probs: dictionary
    :param map: A dictionary {img_id -> [probability distribution]}
    :type map: dictionary
    :return: A list of predictions
    :rtype: list
    """

    size = len(image_ids)
    probssum_map = {}
    for i in range(size):
        image_id = image_ids[i]
        print("image_id: " + image_id)
        probs = probses[i]
        product_id = imageid_to_productid(image_id)

        if product_id in probssum_map:
            probssum_map[product_id] += probs
        else:
            probssum_map[product_id] = probs

    product_to_prediction_map = {}
    for product_id, probs_sum in probssum_map.items():
        prediction = np.argmax(probs_sum.reshape(-1))
        product_to_prediction_map[product_id] = prediction

    # res = {}
    # for i in range(size):
    #     image_id = image_ids[i]
    #     product_id = imageid_to_productid(image_id)
    #     res[image_id] = product_to_prediction_map[product_id]
    #
    # return res

    return product_to_prediction_map

## common functions ##

def get_accuracy(probs, labels, use_cuda):
    probs = probs.cpu().data.numpy() if use_cuda else probs.data.numpy()
    labels = labels.cpu().data if use_cuda else labels.data
    batch_size = probs.shape[0]
    correct_num = 0.0
    for i in range(batch_size):
        index = np.argmax(probs[i].reshape(-1))
        #print("index ",index.numpy()[0])
        #print("label",labels.data[i])
        #print("labels", labels[i])
        #print("indexing:",index)
        if index == labels[i]:
            correct_num = correct_num + 1.0
            #print("correct!")
    return correct_num / batch_size

def save_checkpoint(optimizer, i, epoch, net, best_valid_acc, best_train_acc, train_acc, valid_acc, check_dir, latest_dir, name):
    print("=> Saving checkpoint: " + check_dir + name)

    torch.save({
        'optimizer': optimizer.state_dict(),
        'iter': i,
        'epoch': epoch,
        'state_dict': net.state_dict(),
        'best_valid_acc': best_valid_acc,
        'best_train_acc': best_train_acc,
        'train_acc': train_acc,
        'valid_acc': valid_acc
    }, check_dir + name)
    print("=> Saved checkpoint")
    save_latest(optimizer, i, epoch, net, best_valid_acc, best_train_acc, train_acc, valid_acc, latest_dir)

def save_latest(optimizer, i, epoch, net, best_valid_acc, best_train_acc, train_acc, valid_acc, dir):
    print("=> Update checkpoint: " + dir + "latest.pth")
    torch.save({
        'optimizer': optimizer.state_dict(),
        'iter': i,
        'epoch': epoch,
        'state_dict': net.state_dict(),
        'best_valid_acc': best_valid_acc,
        'best_train_acc': best_train_acc,
        'train_acc': train_acc,
        'valid_acc': valid_acc
    }, dir + "/latest.pth")

    print("=> Updated latest")

def evaluate(net, test_loader, sample_num, use_cuda):

    test_num  = 0
    test_loss = 0
    test_acc  = 0
    cnt = 0

    # for iter, (images, labels, indices) in enumerate(test_loader, 0):
    for iter, (images, labels, _) in enumerate(test_loader, 0):#remove indices for testing
        if test_num > sample_num:
            break

        images = Variable(images.type(torch.FloatTensor)).cuda() if use_cuda else Variable(images.type(torch.FloatTensor))
        labels = Variable(labels).cuda() if use_cuda else Variable(labels)

        logits = net(images)
        probs  = F.softmax(logits)
        #print("labels:", labels)
        #print("probs:",probs)
        loss = F.cross_entropy(logits, labels)
        test_acc += get_accuracy(probs, labels, use_cuda)
        ####
        #acc  = top_accuracy(probs, labels, top_k=(1,))#1,5
        ####
        # batch_size = len(indices)
        batch_size = len(images) # use images instead of indices for testing
        ####
        #test_acc  += batch_size*acc[0][0]
        ####
        test_loss += batch_size*loss.data[0]
        test_num  += batch_size

        cnt = cnt + 1

    test_acc  = test_acc/cnt
    test_loss = test_loss/test_num
    return test_loss, test_acc

def get_gpu_stats():
    from subprocess import call
    command = ["nvidia-smi"]
    call(command)