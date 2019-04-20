import pandas as pd
from scipy import stats
import tqdm as tqdm
from Utils import *
# # Generate final data csv
# csv_file = "./data/train_data.csv"
# data_csv = pd.read_csv(csv_file)
# products = list(data_csv['product_id'])
# internal_ids = list(data_csv['image_id'])
# labels = list(data_csv['category_id'])
# num = len(products)
#
# rows = {}
# for i in range(num):
#     external_image_id = '{}-{}'.format(products[i], internal_ids[i])
#     row = [external_image_id, labels[i]]
#     rows[i] = row
#
# columns = ["image_id", "category_id"]
# df = pd.DataFrame.from_dict(rows, orient="index")
# df.index.name = "index"
# df.columns = columns
# df.sort_index(inplace=True)
# df.to_csv("./data/data.csv")

# Split data into training/validation evenly

# ## get data stats
# csv_file = "./data/data.csv"
# data_csv = pd.read_csv(csv_file)
# external_ids = list(data_csv['image_id'])
# labels = list(data_csv['category_id'])
# num = len(external_ids)
# labels_set = set(labels) # distinct cate
#
# map = {} # cate_id -> [image_id]
# for i in range(num):
#     if labels[i] in map:
#         map[labels[i]].append(external_ids[i])
#     else:
#         map[labels[i]] = [external_ids[i]]
#
# class_num = len(map) # number of classes
# # get stats of number of images in each class
# num_map = {cate:len(map[cate]) for cate in map} # cate_id -> num
# nums = num_map.values()
# stats.describe(nums)
#
# ## split into train and validation
# val_num_2000 = 200 # for those classes with images [2000, +)
# val_r_1000 = 0.05 # for those classes with images [1000, 2000)
# val_r_100 = 0.02 # for those classes with images [100, 1000)
# val_num_0 = 2 # for those classes with images [0, 100)
#
# val_rows = {}
# train_rows = {}
# val_cnt = 0
# train_cnt = 0
# for cate_id, images in tqdm(map.items()):
#     val_images = []
#     train_images = []
#     split = 0
#     if num_map[cate_id] >= 2000:
#         split = val_num_2000
#     elif num_map[cate_id] >= 1000:
#         split = int(num_map[cate_id] * val_r_1000)
#     elif num_map[cate_id] >= 100:
#         split = int(num_map[cate_id] * val_r_100)
#     else:
#         split = val_num_0
#
#     val_images = images[0:split]
#     train_images = images[split:]
#
#     for val_img in val_images:
#         val_row = [val_img, cate_id]
#         val_rows[val_cnt] = val_row
#         val_cnt = val_cnt + 1
#
#     for train_img in train_images:
#         train_row = [train_img, cate_id]
#         train_rows[train_cnt] = train_row
#         train_cnt = train_cnt + 1

# # 252934
# columns = ["image_id", "category_id"]
# df = pd.DataFrame.from_dict(val_rows, orient="index")
# df.index.name = "index"
# df.columns = columns
# df.sort_index(inplace=True)
# df.to_csv("./data/validation.csv")

csv_file = "./data/validation.csv"
data_csv = pd.read_csv(csv_file)
external_ids = list(data_csv['image_id'])
labels = list(data_csv['category_id'])
num = len(external_ids)
labels_set = []

for image_id in external_ids:
    product_id = imageid_to_productid(image_id)
    labels_set.append(product_id)

labels_set = set(labels_set)  # distinct cate
print "validation set product num: ", len(labels_set)

csv_file = "./data/train.csv"
data_csv = pd.read_csv(csv_file)
external_ids = list(data_csv['image_id'])
labels = list(data_csv['category_id'])
num = len(external_ids)
train_labels_set = []

for image_id in external_ids:
    product_id = imageid_to_productid(image_id)
    train_labels_set.append(product_id)

train_labels_set = set(train_labels_set)  # distinct cate
print "train set product num: ", len(train_labels_set)

# # 12118359
# df = pd.DataFrame.from_diwct(train_rows, orient="index")
# df.index.name = "index"
# df.columns = columns
# df.sort_index(inplace=True)
# df.to_csv("./data/train.csv")

# ## split into train and validation
# val_num_2000 = 100 # for those classes with images [2000, +)
# val_r_1000 = 0.03 # for those classes with images [1000, 2000)
# val_r_100 = 0.02 # for those classes with images [100, 1000)
# val_num_0 = 3 # for those classes with images [0, 100)
#
# val_rows = {}
# train_rows = {}
# val_cnt = 0
# train_cnt = 0
# for cate_id, images in tqdm(map.items()):
#     val_images = []
#     train_images = []
#     split = 0
#     if num_map[cate_id] >= 2000:
#         split = val_num_2000
#     elif num_map[cate_id] >= 1000:
#         split = int(num_map[cate_id] * val_r_1000)
#     elif num_map[cate_id] >= 100:
#         split = int(num_map[cate_id] * val_r_100)
#     else:
#         split = val_num_0
#
#     val_images = images[0:split]
#     train_images = images[split:]
#
#     for val_img in val_images:
#         val_row = [val_img, cate_id]
#         val_rows[val_cnt] = val_row
#         val_cnt = val_cnt + 1
#
#     for train_img in train_images:
#         train_row = [train_img, cate_id]
#         train_rows[train_cnt] = train_row
#         train_cnt = train_cnt + 1
#
# # 141940
# columns = ["image_id", "category_id"]
# df = pd.DataFrame.from_dict(val_rows, orient="index")
# df.index.name = "index"
# df.columns = columns
# df.sort_index(inplace=True)
# df.to_csv("./data/validation_small.csv")