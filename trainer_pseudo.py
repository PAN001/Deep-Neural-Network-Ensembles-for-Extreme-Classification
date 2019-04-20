from __future__ import print_function

import os
from datetime import *
from torch.utils.data.sampler import RandomSampler
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from timeit import default_timer as timer
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time

# custom modules
from Log import *
from StepLR import *
from StepAlpha import *
from Utils import *
from AverageMeter import *
from cdimage import *

# --------------------------------------------------------
from net.resnet101 import ResNet101 as Net
# from net.excited_inception_v3 import SEInception3 as Net
# from net.xception import Xception as Net


IDENTIFIER = "resnet_pseudo"

# Not change
use_cuda = True
SEED = 123456
PROJECT_PATH = './project'
CDISCOUNT_HEIGHT = 180
CDISCOUNT_WIDTH = 180
CDISCOUNT_NUM_CLASSES = 5270

# Dirs/Paths
out_dir  = '../'
csv_dir = './data/'
root_dir = '../output/'
train_data_filename = 'train.csv'
validation_data_filename = 'validation.csv'
train_pseudo_data_filename = "test_pseudo_labeled.csv"
checkpoint_dir = "../checkpoint/" + IDENTIFIER + "/"
latest_dir = "./latest/" + IDENTIFIER + "/"
log_dir = "./log/" + IDENTIFIER + "/"

## Create output folder
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(latest_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(out_dir +'/backup/' + IDENTIFIER, exist_ok=True)

log = Log(log_dir + "log.out")



def run_training():

    #-------------------------------------------- Training settings --------------------------------------------

    # initial_checkpoint = None
    initial_checkpoint = "./latest/resnet/latest.pth"
    # initial_checkpoint = latest_dir + "latest.pth"
    # initial_checkpoint = '../trained_models/resnet_00243000_model.pth'
    # pretrained_file = '../trained_models/LB=0.69422_xception_00158000_model.pth'
    pretrained_file = None
    skip = [] #['fc.weight', 'fc.bias']

    num_iters   = 1000*1000
    iter_smooth = 50
    iter_valid  = 800
    iter_log = 5 # i
    iter_latest = 50
    iter_save_freq = 1000 # i
    iter_save   = [0, num_iters-1] + list(range(0,num_iters,1*iter_save_freq)) # first and last iters, then every 1000 iters
    iter_org_train = 3 # every 3 iters(j) of pseudo training set, use 1 iter of original train

    validation_num = 10000

    # Hyper parameters
    pseudo_alpha = 0.3
    batch_size  = 64 #60   #512  #96 #256
    pseudo_batch_size = 64
    validation_batch_size = 64
    iter_accum  = 3 # j

    batch_loss  = 0.0
    batch_acc   = 0.0
    best_valid_acc  = 0.0
    best_train_acc  = 0.0
    rate = 0

    iter_time_meter = AverageMeter()

    train_loss_meter = AverageMeter()
    train_acc_meter = AverageMeter()

    org_train_loss_meter = AverageMeter()
    org_train_acc_meter = AverageMeter()

    valid_loss_meter = AverageMeter()
    valid_acc_meter = AverageMeter()

    j = 0 # number of iters in total
    i = 0 # number of real iters where bp is conducted

    #-----------------------------------------------------------------------------------------------------------

    log.write('\n--- [START %s] %s\n\n' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 64))
    log.write('** some experiment setting **\n')
    log.write('\tIDENTIFIER   = %s\n' % IDENTIFIER)
    log.write('\tSEED         = %u\n' % SEED)
    log.write('\tPROJECT_PATH = %s\n' % PROJECT_PATH)
    log.write('\tout_dir      = %s\n' % out_dir)

    ## net -------------------------------
    log.write('** net setting **\n')
    print("=> Initing net ...")
    net = Net(in_shape = (3, CDISCOUNT_HEIGHT, CDISCOUNT_WIDTH), num_classes=CDISCOUNT_NUM_CLASSES)
    if use_cuda: net.cuda()
    print("=> Inited net ...")
    get_gpu_stats()
    ####
    # if 0: #freeze early layers
    #     for p in net.layer0.parameters():
    #         p.requires_grad = False
    #     for p in net.layer1.parameters():
    #         p.requires_grad = False
    #     for p in net.layer2.parameters():
    #         p.requires_grad = False
    #     for p in net.layer3.parameters():
    #         p.requires_grad = False

    log.write('%s\n\n'%(type(net)))
    # log.write('\n%s\n'%(str(net)))
    # log.write(inspect.getsource(net.__init__)+'\n')
    # log.write(inspect.getsource(net.forward )+'\n')
    log.write('\n')

    ## optimiser ----------------------------------
    #LR = StepLR([ (0, 0.01),  (200, 0.001),  (300, -1)])
    LR = StepLR([ (0, 0.01), (1, 0.005), (3, 0.0005)])
    Alpha = StepAlpha()

    ## optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)  ###0.0005
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.01, momentum=0.1, weight_decay=0.0001)

    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')
    print("=> Initing training set ...")
    transform_train = transforms.Compose([transforms.Lambda(lambda x: net.train_augment(x))])
    train_dataset = CDiscountDataset(csv_dir+train_data_filename,root_dir,"train",transform=transform_train)
    train_loader  = DataLoader(
                        train_dataset,
                        sampler = RandomSampler(train_dataset),
                        batch_size  = batch_size,
                        drop_last   = True,
                        num_workers = 1,
                        pin_memory  = False)
    train_loader_iter = iter(train_loader)
    print("=> Inited training set")
    get_gpu_stats()

    print("=> Initing pseudo training set ...")
    pseudo_train_dataset = CDiscountDataset(csv_dir+train_pseudo_data_filename,root_dir,"test",transform=transform_train)
    pseudo_train_loader  = DataLoader(
        pseudo_train_dataset,
                        sampler = RandomSampler(pseudo_train_dataset),
                        batch_size  = pseudo_batch_size,
                        drop_last   = True,
                        num_workers = 8,
                        pin_memory  = False)
    print("=> Inited pseudo training set")
    get_gpu_stats()

    print("=> Initing validation set ...")
    transform_valid = transforms.Compose([transforms.Lambda(lambda x: net.valid_augment(x))])
    valid_dataset = CDiscountDataset(csv_dir+validation_data_filename,root_dir,"train",transform=transform_valid)
    valid_loader  = DataLoader(
                        valid_dataset,
                        sampler     = SequentialSampler(valid_dataset),
                        batch_size  = validation_batch_size,
                        drop_last   = False,
                        num_workers = 8,
                        pin_memory  = False)
    print("=> Inited validation set")
    get_gpu_stats()

    log.write('\tlen(train_dataset)  = %d\n'%(len(train_dataset)))
    log.write('\tlen(valid_dataset)  = %d\n'%(len(valid_dataset)))
    log.write('\tlen(train_loader)   = %d\n'%(len(train_loader)))
    log.write('\tlen(valid_loadernum_iters)   = %d\n'%(len(valid_loader)))
    log.write('\tbatch_size  = %d\n'%(batch_size))
    log.write('\titer_accum  = %d\n'%(iter_accum))
    log.write('\tbatch_size*iter_accum  = %d\n'%(batch_size*iter_accum))

    ## resume from previous ----------------------------------
    start_iter = 0
    start_epoch= 0.
    if initial_checkpoint is not None: # load a checkpoint and resume from previous training
        log.write('\tloading @ initial_checkpoint = %s\n' % initial_checkpoint)

        # load
        if os.path.isfile(initial_checkpoint):
            print("=> loading checkpoint '{}'".format(initial_checkpoint))

            checkpoint = torch.load(initial_checkpoint)

            start_epoch = checkpoint['epoch']
            start_iter = checkpoint['iter']
            best_train_acc = checkpoint['best_train_acc']
            best_valid_acc = checkpoint['best_valid_acc']
            train_acc_meter.update(checkpoint['train_acc'])
            valid_acc_meter.update(checkpoint['valid_acc'])
            net.load_state_dict(checkpoint['state_dict'])  # load model weights from the checkpoint
            optimizer.load_state_dict(checkpoint['optimizer'])

            # # load original checkpoint
            # net.load_state_dict(checkpoint)

            log.write("=> loaded checkpoint '{}' (epoch: {}, iter: {}, best_train_acc: {}, best_valid_acc: {})"
                  .format(initial_checkpoint, start_epoch, start_iter, best_train_acc, best_valid_acc))
            get_gpu_stats()

            # # load original model
            # log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
            # net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))
            #
            # checkpoint = torch.load(initial_checkpoint.replace('_model.pth', '_optimizer.pth'))
            # start_iter = checkpoint['iter']
            # start_epoch = checkpoint['epoch']

        else:
            print("=> no checkpoint found at '{}'".format(initial_checkpoint))
            exit(0)
    elif pretrained_file is not None: # load a pretrained model and train from the beginning
        log.write('\tloading @ pretrained_file = %s\n' % pretrained_file)
        net.load_pretrain_pytorch_file( pretrained_file, skip)


    ## start training here! ##############################################
    log.write('** start training here! **\n')

    log.write('\toptimizer=%s\n'%str(optimizer) )
    # log.write(' LR=%s\n\n'%str(LR) )
    log.write('   rate   alpha   iter   epoch  | valid_loss/acc | train_loss/acc | org_train_loss/acc | batch_loss/acc | total time | avg iter time | i j |\n')
    log.write('--------------------------------------------------------------------------------------------------------------------------------------------\n')

    # Custom setting
    start_iter = 0
    i = start_iter
    start_epoch= start_iter*pseudo_batch_size*iter_accum/len(pseudo_train_dataset)

    start = timer()
    end = time.time()
    while  i<num_iters:
        net.train()
        optimizer.zero_grad()

        log.write("=> New epoch for pseudo training set\n")

        for k, data in enumerate(pseudo_train_loader, 0):
            images,labels,_ = data

            i = j/iter_accum + start_iter
            epoch = (i-start_iter)*pseudo_batch_size*iter_accum/len(pseudo_train_dataset) + start_epoch

            if i % iter_log == 0:
                # print('\r',end='',flush=True)
                log.write('\r%0.4f  %0.4f  %5.1f k   %4.2f  | %0.4f  %0.4f | %0.4f  %0.4f | %0.4f  %0.4f | %0.4f  %0.4f | %5.0f min | %5.2f s | %d,%d \n' % \
                        (rate, pseudo_alpha, i/1000, epoch, valid_loss_meter.val, valid_acc_meter.val, train_loss_meter.avg, train_acc_meter.avg, org_train_loss_meter.avg, org_train_acc_meter.avg, batch_loss, batch_acc,(timer() - start)/60,
                            iter_time_meter.avg, i, j))

            if i in iter_save and i != start_iter:
                save_checkpoint(optimizer, i, epoch, net, best_valid_acc, best_train_acc, train_acc_meter.val, valid_acc_meter.val, checkpoint_dir, latest_dir, "%08d_model.pth"%(i))

            if i % iter_latest == 0 and i != start_iter:
                save_latest(optimizer, i, epoch, net, best_valid_acc, best_train_acc, train_acc_meter.val, valid_acc_meter.val, latest_dir)

            if i % iter_valid == 0 and i != start_iter:
                print("\n=> Validating ...")
                net.eval()
                loss, acc = evaluate(net, valid_loader, validation_num, use_cuda)
                valid_loss_meter.update(loss)
                valid_acc_meter.update(acc)
                net.train()
                print("\n=> Validated")

                # update best valida_acc and update best model
                if valid_acc_meter.val > best_valid_acc:
                    best_valid_acc = valid_acc_meter.val

                    # update best model on validation set
                    # torch.save(net.state_dict(), out_dir + '/checkpoint/best_model.pth')
                    save_checkpoint(optimizer, i, epoch, net, best_valid_acc, best_train_acc, train_acc_meter.val, valid_acc_meter.val, checkpoint_dir, latest_dir, "best_val_model.pth")
                    log.write("=> Best validation model updated: iter %d, validation acc %f\n" % (i, best_valid_acc))

            # learning rate/alpha schduler -------------
            lr = LR.get_rate(epoch)
            pseudo_alpha = Alpha.get_rate(epoch)
            if lr<0 : break
            adjust_learning_rate(optimizer, lr/iter_accum)
            rate = get_learning_rate(optimizer)[0]*iter_accum

            end = time.time()
            # one iteration update  -------------
            images = Variable(images.type(torch.FloatTensor)).cuda() if use_cuda else Variable(images.type(torch.FloatTensor))
            labels = Variable(labels).cuda() if use_cuda else Variable(labels)
            logits = net(images)
            probs  = F.softmax(logits)
            loss = pseudo_alpha * F.cross_entropy(logits, labels)
            batch_loss = loss.data[0]
            train_loss_meter.update(batch_loss)
            batch_acc  = get_accuracy(probs, labels, use_cuda)
            train_acc_meter.update(batch_acc)

            # accumulate gradients
            loss.backward()

            if j%iter_org_train == 0:
                # print("\n=> use org train data")
                cur = next(train_loader_iter, None)
                if(cur == None):
                    train_loader_iter = iter(train_loader) # restart from the beginning
                    cur = next(train_loader_iter, None)
                    assert(cur != None)
                org_images = cur[0]
                org_images = Variable(org_images.type(torch.FloatTensor)).cuda() if use_cuda else Variable(org_images.type(torch.FloatTensor))
                org_labels = cur[1]
                org_labels = Variable(org_labels).cuda() if use_cuda else Variable(org_labels)
                org_logits = net(org_images)
                org_probs = F.softmax(org_logits)
                org_loss = F.cross_entropy(org_logits, org_labels)
                batch_loss = org_loss.data[0]
                org_train_loss_meter.update(batch_loss)
                batch_acc = get_accuracy(org_probs, org_labels, use_cuda)
                org_train_acc_meter.update(batch_acc)

                # accumulate gradients
                org_loss.backward()

            # update gradients every iter_accum
            if j%iter_accum == 0:
                #torch.nn.utils.clip_grad_norm(net.parameters(), 1)
                # print("\n=> optim step")
                optimizer.step()
                optimizer.zero_grad()

            # measure elapsed time
            iter_time_meter.update(time.time() - end)

            # print statistics  ------------
            if i%iter_smooth == 0 and i != start_iter: # reset train stats every iter_smooth iters
                if train_acc_meter.avg > best_train_acc:
                    best_train_acc = train_acc_meter.avg
                    # update best model on train set
                    save_checkpoint(optimizer, i, epoch, net, best_valid_acc, best_train_acc, train_acc_meter.val, valid_acc_meter.val, checkpoint_dir, latest_dir, "best_train_model.pth")
                    log.write("=> Best train model updated: iter %d, train acc %f\n"%(i, best_train_acc))

                train_loss_meter = AverageMeter()
                train_acc_meter = AverageMeter()

            print('\r%0.4f  %0.4f  %5.1f k   %4.2f  | %0.4f  %0.4f | %0.4f  %0.4f | %0.4f  %0.4f | %0.4f  %0.4f | %5.0f min | %5.2f s | %d,%d' % \
                    (rate, pseudo_alpha, i/1000, epoch, valid_loss_meter.val, valid_acc_meter.val, train_loss_meter.avg, train_acc_meter.avg, org_train_loss_meter.avg, org_train_acc_meter.avg, batch_loss, batch_acc, (timer() - start)/60, iter_time_meter.avg, i, j),\
                    end='',flush=True)
            j=j+1
        pass  #-- end of one data loader --
    pass #-- end of all iterations --


    ## check : load model and re-test
    if 1:
        save_checkpoint(optimizer, i, epoch, net, best_valid_acc, best_train_acc, train_acc_meter.val, valid_acc_meter.val, checkpoint_dir, latest_dir, "%d_optimizer.pth"%(i))

    log.write('\n')

##to determine best threshold etc ... ## ------------------------------


# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_training()


    print('\nsucess!')
