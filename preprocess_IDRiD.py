import os
import pandas as pd

source_path = '/home/zouwei/dataset/IDRiD/Disease_Grading/'
image_folder = source_path + 'Original_Images/'
label_path = source_path + 'Groundtruths/'

train = []
test = []

f_train = pd.read_csv(label_path + 'a. IDRiD_Disease Grading_Training Labels.csv')
for i in range(len(f_train)):
    image_path = image_folder + 'Training_Set/' + str(f_train.iat[i, 0]) + '.jpg'
    label_DR = str(f_train.iat[i, 1])
    label_DME = str(f_train.iat[i, 2])
    train.append(image_path + ' ' + label_DR + ' ' + label_DME + '\n')

f = open(source_path + 'dataset_train.txt', 'w')
for line in train:
    f.write(line)
f.close()

f_test = pd.read_csv(label_path + 'b. IDRiD_Disease Grading_Testing Labels.csv')
for i in range(len(f_test)):
    image_path = image_folder + 'Testing_Set/' + str(f_test.iat[i, 0]) + '.jpg'
    label_DR = str(f_test.iat[i, 1])
    label_DME = str(f_test.iat[i, 2])
    test.append(image_path + ' ' + label_DR + ' ' + label_DME + '\n')

f = open(source_path + 'dataset_test.txt', 'w')
for line in test:
    f.write(line)
f.close()
    f.write(line)
f.close()