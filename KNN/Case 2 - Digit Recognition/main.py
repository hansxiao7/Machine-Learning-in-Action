import os
import numpy as np

test_dir = 'testDigits/'
train_dir = 'trainingDigits/'
n_dimension = 32

#load training data and label, create validation set
train_data = []
for train_file in os.listdir(train_dir):
    data = np.loadtxt(train_dir+train_file, dtype=str)
    int_data = []
    for i in range(np.shape(data)[0]):
        row_list = []
        for j in range(n_dimension):
            row_list.append(int(data[i][j]))
        int_data.append(row_list)
    int_data = np.array(int_data)
    int_data = int_data.flatten() #put data to 1-D from 2-D
    label = int(train_file.split('_')[0])
    train_data.append([int_data, label])

n_data = len(train_data)

np.random.shuffle(train_data)
train_ratio = 0.7

n_train = int(train_ratio * n_data)
n_val = n_data - n_train

train_set = []
train_label = []
for i in range(n_train):
    train_set.append(train_data[i][0])
    train_label.append(train_data[i][1])

val_set = []
val_label = []
for j in range(n_val):
    val_set.append(train_data[j+n_train][0])
    val_label.append(train_data[j+n_train][1])

#load test data and label
test_data = []
for test_file in os.listdir(test_dir):
    data = np.loadtxt(test_dir+test_file, dtype=str)
    int_data = []
    for i in range(np.shape(data)[0]):
        row_list = []
        for j in range(n_dimension):
            row_list.append(int(data[i][j]))
        int_data.append(row_list)
    int_data = np.array(int_data)
    int_data = int_data.flatten() #put data to 1-D from 2-D
    label = int(test_file.split('_')[0])
    test_data.append([int_data, label])

n_test = len(test_data)

test_set = []
test_label = []
for i in range(n_test):
    test_set.append(test_data[i][0])
    test_label.append(test_data[i][1])


#since the data is all 0 and 1, no normalization is necessary
#define k-NN
def kNN(test_data, test_label, train_data, train_label, k):
    n_train = np.shape(train_data)[0]
    n_test = np.shape(test_data)[0]

    esti_result = []
    error = 0
    for i in range(n_test):
        x = test_data[i]
        x_array = np.tile(x, (n_train, 1))
        diff = x_array - train_data
        diff = diff ** 2
        dist = np.sqrt(np.sum(diff, axis=1))
        dist_order_list = np.argsort(dist)

        label_esti_dict = {}
        for j in range(k):
            esti_label = train_label[dist_order_list[j]]
            if label_esti_dict.get(esti_label) is None:
                label_esti_dict[esti_label] = 0

            label_esti_dict[esti_label] += 1

        dict_keys = list(label_esti_dict.keys())
        dict_values = list(label_esti_dict.values())

        result = dict_keys[dict_values.index(max(dict_values))]
        esti_result.append(result)

        if result != test_label[i]:
            error += 1

    return esti_result, error

k = 3
esti_labels_val, error_val = kNN(val_set, val_label, train_set, train_label, k)
esti_labels_test, error_test = kNN(test_set, test_label, train_set, train_label, k)
print('For the validation data set, total validation error is:'+'\n'+str(error_val/n_val))
print('For the test data set, total test error is:'+'\n'+str(error_test/n_test))