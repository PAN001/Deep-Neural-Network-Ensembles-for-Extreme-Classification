from label_category_transform import *
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data.sampler import SequentialSampler
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from transform import *
import torch
import random

csv_dir = './data/'
root_dir = '../output/train/'
data_file_name = 'train_data.csv'


class CDiscountDataset(Dataset):
    def __init__(self, csv_dir, root_dir, mode = "train", transform=None):
        # print("loading CDiscount Dataset...")
        self.image_names=[]
        self.root_dir=root_dir
        self.transform = transform
        self.mode = mode
        image_data = pd.read_csv(csv_dir)
        self.image_id = list(image_data['image_id'])
        if self.mode == "train" or self.mode == "valid":
            self.labels = list(image_data['category_id'])
            self.indexes = list(image_data['category_id'])
        num_train = len(image_data)
        for i in range(num_train):
            if self.mode == "train" or self.mode == "valid":
                self.indexes[i] = category_id_to_index[self.labels[i]]
                image_name = '{}/{}.jpg'.format(self.labels[i],self.image_id[i])
            elif self.mode == "test":
                image_name = '{}.jpg'.format(self.image_id[i])
            else:
                print("mode should be : train/valid/test")
                exit()
            self.image_names.append(image_name)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        if self.mode == "train" or self.mode == "valid":
            img = cv2.imread(self.root_dir + 'train/'+ self.image_names[idx])
        else:
            img = cv2.imread(self.root_dir + 'test/' + self.image_names[idx])
        label = []
        if self.mode == "train" or self.mode == "valid":
            label = self.indexes[idx]
        img_id = self.image_id[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label, img_id








