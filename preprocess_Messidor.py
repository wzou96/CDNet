import os
import pandas as pd

source_path = '/home/zouwei/dataset/Messidor/'
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

f = open(source_path + 'dataset.txt', 'w')
for line in dataset:
    f.write(line)
f.close()

txt_path = source_path + '10fold/'
for root, dirs, files in os.walk(txt_path):
    for file in files:
        dataset_fold_test = []
        f = open(txt_path + file, 'r')
        for line in f.readlines():
            dataset_fold_test.append(dataset[int(line.strip())])
        f.close()
        dataset_fold_train = list(set(dataset).difference(set(dataset_fold_test)))
        dataset_fold_train.sort()

        f_fold_train = open(source_path + 'dataset_' + file.split('.')[0] + '_train.txt', 'w')
        for line in dataset_fold_train:
            f_fold_train.write(line)
        f_fold_train.close()

        f_fold_test = open(source_path + 'dataset_' + file.split('.')[0] + '_test.txt', 'w')
        for line in dataset_fold_test:
            f_fold_test.write(line)
        f_fold_test.close()
)
