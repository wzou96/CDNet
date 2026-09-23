import os
import pandas as pd

source_path = '/home/zouwei/dataset/Messidor/'
save_path = '/home/zouwei/dataset/'

list_excel = ['Base11', 'Base12', 'Base13', 'Base14',
        'Base21', 'Base22', 'Base23','Base24',
        'Base31', 'Base32', 'Base33', 'Base34']

dataset = []

for sublist in list_excel:
    file_path = source_path + 'Annotation_' + sublist + '.xls'
    data = pd.read_excel(file_path)
    for i in range(len(data)):
        image_path = source_path + sublist + '/' + str(data.iat[i, 0])
        label_DR = str(data.iat[i, 2])
        label_DME = str(data.iat[i,3])
        dataset.append(image_path + ' ' + label_DR + ' ' + label_DME + '\n')

f = open(save_path + 'dataset_cross_train.txt', 'w')
for line in dataset:
    f.write(line)
f.close()

source_path = '/home/zouwei/dataset/IDRiD/Disease_Grading/'
image_folder = source_path + 'Original_Images/'
label_path = source_path + 'Groundtruths/'

dataset = []

f_train = pd.read_csv(label_path + 'a. IDRiD_Disease Grading_Training Labels.csv')
for i in range(len(f_train)):
    image_path = image_folder + 'Training_Set/' + str(f_train.iat[i, 0]) + '.jpg'
    label_DR = f_train.iat[i, 1]
    if label_DR ==4:
        label_DR = 3
    label_DME = f_train.iat[i, 2]
    dataset.append(image_path + ' ' + str(label_DR) + ' ' + str(label_DME) + '\n')


f_test = pd.read_csv(label_path + 'b. IDRiD_Disease Grading_Testing Labels.csv')
for i in range(len(f_test)):
    image_path = image_folder + 'Testing_Set/' + str(f_test.iat[i, 0]) + '.jpg'
    label_DR = f_test.iat[i, 1]
    if label_DR ==4:
        label_DR = 3
    label_DME = f_test.iat[i, 2]
    dataset.append(image_path + ' ' + str(label_DR) + ' ' + str(label_DME) + '\n')

f = open(save_path + 'dataset_cross_test.txt', 'w')
for line in dataset:
    f.write(line)
f.close()