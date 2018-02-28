# Author: Sebastian Fabig
import os
import csv

train_data = './ProstateData/TrainImages'
train_mask_data = './ProstateData/MaskTrainOne'
test_data = './ProstateData/TestImages'
test_mask_data = './ProstateData/MaskTestOne'

train_files = os.listdir(train_data)
train_masks = os.listdir(train_mask_data)
test_files = os.listdir(test_data)
test_masks = os.listdir(test_mask_data)

csv_list = list()
csv_list.append(['img', 'mask'])
for i in range(len(train_files)):
    csv_list.append([os.path.relpath(train_data + "/" + train_files[i]), os.path.relpath(train_mask_data + "/" + train_masks[i])])

with open('./prostate_training.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(csv_list)

csv_list = list()
csv_list.append(['img', 'mask'])
for i in range(len(test_files)):
    csv_list.append([os.path.relpath(test_data + "/" + test_files[i]),
                     os.path.relpath(test_mask_data + "/" + test_masks[i])])

with open('./prostate_test.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(csv_list)

print(len(train_files))
print(len(train_masks))