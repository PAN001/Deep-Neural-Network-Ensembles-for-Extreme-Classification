import os
from torch.autograd import Variable
import torch.nn.functional as F
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from transform import *
from Utils import *
from cdimage import *
from torch.utils.data.sampler import RandomSampler
import operator



from net.resnet101 import ResNet101 as Net




use_cuda = False
IDENTIFIER = "resnet"
SEED = 123456
PROJECT_PATH = './project'
CDISCOUNT_HEIGHT = 180
CDISCOUNT_WIDTH = 180
CDISCOUNT_NUM_CLASSES = 5270

csv_dir = './data/'
root_dir = '../output/'
test_data_filename = 'test.csv'
validation_data_filename = 'validation.csv'
validation_batch_size = 128


transform_valid = transforms.Compose([transforms.Lambda(lambda x: general_valid_augment(x))])

test_dataset = CDiscountTestDataset(csv_dir + test_data_filename, root_dir, transform=transform_valid)

test_loader = DataLoader(
    test_dataset,
    sampler=SequentialSampler(test_dataset),
    batch_size=validation_batch_size,
    drop_last=False,
    num_workers=0,
    pin_memory=False)

for iter, (images, image_ids) in enumerate(test_loader, 0):
    images = Variable(images.type(torch.FloatTensor)).cuda() if use_cuda else Variable(images.type(torch.FloatTensor))
    image_ids = np.array(image_ids)
    print(image_ids)
    print("-----------------------------------------------------------------------------------")

