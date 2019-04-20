from label_category_transform import *
import pandas as pd





csv_dir = "../test_res/resnet_test.res"
path = "../test_res/resnet_result.csv"
product_dic = []

category_id = []
cnt = 0

data = pd.read_csv(csv_dir)
product_id = list(data['_id'])
class_id = list(data['category_id'])
rows = {}
num = len(product_id)

with open(path, "a") as file:
    file.write("_id,category_id\n")

    for i in range(num):
        print(product_id[i])
        print(label_to_category_id[class_id[i]])
        file.write(str(product_id[i]) + "," + str(label_to_category_id[class_id[i]]) + "\n")


