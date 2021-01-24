# k-NN for Case 1
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def load_data(filename):
    data = np.loadtxt(filename, dtype = 'str')
    np.random.shuffle(data)

    train_ratio = 0.7
    n_train = int(np.shape(data)[0] * train_ratio)
    n_val = np.shape(data)[0] - n_train

    train_label = data[0:n_train, -1]
    train_data = data[0:n_train, 0:-1]
    train_data = train_data.astype('float64')

    val_label = data[n_train:np.shape(data)[0], -1]
    val_data = data[n_train:np.shape(data)[0], 0:-1]
    val_data = val_data.astype('float64')

    return train_data, train_label, val_data, val_label


def figure_plot(data, label):
    fig, (f1, f2, f3) = plt.subplots(3, 1)

    # find label locations
    class_1_loc = np.where(label == 1)[0]
    class_2_loc = np.where(label == 2)[0]
    class_3_loc = np.where(label == 3)[0]
    class_loc = [class_1_loc, class_2_loc, class_3_loc]

    colors = ['r', 'g', 'b']

    x1 = data[:, 0]
    x2 = data[:, 1]
    x3 = data[:, 2]

    for i in range(3):
        f1.scatter(x1[class_loc[i]], x2[class_loc[i]], c=colors[i], label='class' + str(i + 1))
        f2.scatter(x1[class_loc[i]], x3[class_loc[i]], c=colors[i], label='class' + str(i + 1))
        f3.scatter(x2[class_loc[i]], x3[class_loc[i]], c=colors[i], label='class' + str(i + 1))

    f1.set_xlabel('x1')
    f1.set_ylabel('x2')
    f2.set_xlabel('x1')
    f2.set_ylabel('x3')
    f3.set_xlabel('x2')
    f3.set_ylabel('x3')

    plt.show()


def data_normalization(data):
    min_value = np.amin(data, axis=0)
    max_value = np.amax(data, axis=0)

    n_set = np.shape(data)[0]

    min_matrix = np.tile(min_value, (n_set, 1))
    max_matrix = np.tile(max_value, (n_set, 1))

    normalized_data = (data - min_matrix) / (max_matrix - min_matrix)

    return normalized_data, min_value, max_value


def knn(test_data, test_label, data, labels, k):
    # X is the validation/test cases
    n = np.shape(data)[0]

    results = []
    for x in test_data:
        x_array = np.tile(x, (n, 1))
        diff = x_array - data
        diff = diff ** 2
        dist = np.sqrt(np.sum(diff, axis=1))
        dist_sort_loc = np.argsort(dist)

        estimated_label_dict = {}
        for i in range(k):
            estimated_label = labels[dist_sort_loc[i]]
            if estimated_label_dict.get(estimated_label) is None:
                estimated_label_dict[estimated_label] = 0
            estimated_label_dict[estimated_label] += 1

        keys = list(estimated_label_dict.keys())
        values = list(estimated_label_dict.values())

        results.append(keys[values.index(max(values))])
    n_test = np.shape(test_label)[0]
    error = 0
    for i in range(n_test):
        if test_label[i] != results[i]:
            error += 1

    return error, n_test, results


filename = 'datingTestSet.txt'
train_data, train_label, val_data, val_label = load_data(filename)
train_data, min_value, max_value = data_normalization(train_data)

min_matrix = np.tile(min_value, (np.shape(val_data)[0], 1))
max_matrix = np.tile(max_value, (np.shape(val_data)[0], 1))

normalized_val_data = (val_data - min_matrix) / (max_matrix - min_matrix)

k = 3
error, n_test, results = knn(normalized_val_data, val_label, train_data, train_label, k)
print('The error in the validation data set is' + '\n' + str(error/n_test))
# print("Based on this guy's information, Helen may "+results[0])
