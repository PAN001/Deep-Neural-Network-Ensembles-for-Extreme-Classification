import pandas as pd

csv_dir = "./data/test.csv"

product_dic = []

pre_id = ""
cnt = 0

image_data = pd.read_csv(csv_dir)
image_id = list(image_data['image_id'])
image_num = len(image_data)
print("image num:", image_num)
for i in range(image_num):
    if i % 5000 == 0:
        print(i,"/",image_num)
    line = image_id[i].split("-")
    product_id = line[0]
    if product_id == pre_id:continue
    cnt = cnt +1
    pre_id = product_id
print("product num:", cnt)