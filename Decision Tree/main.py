import numpy as np
import os


def calculate_entropy(data_set):
    data_dict = {}
    for data in data_set:
        y = data[-1]
        if data_dict.get(y) is None:
            data_dict[y] = 0
        data_dict[y] += 1

    frequency = list(data_dict.values())

    total_sum = np.sum(frequency)
    frequency = frequency / total_sum
    frequency = frequency * np.log2(frequency)

    entropy = -np.sum(frequency)

    return entropy


def split_data_set(data_set, considered_para, para_names):
    curr_entropy = calculate_entropy(data_set)
    tree = {}

    if curr_entropy == 0:
        return data_set[0][-1]

    split_para = None
    output_data_dict = {}
    min_entropy = 0
    for i in considered_para:
        trial_data_dict = {}
        for j in range(len(data_set)):
            if trial_data_dict.get(data_set[j][i]) is None:
                trial_data_dict[data_set[j][i]] = []
            trial_data_dict[data_set[j][i]].append(data_set[j])

        sub_entropy = 0
        for key in list(trial_data_dict.keys()):
            sub_entropy += calculate_entropy(trial_data_dict[key])

        if min_entropy == 0 or sub_entropy < min_entropy:
            split_para = i
            min_entropy = sub_entropy
            output_data_dict = trial_data_dict.copy()
    if split_para is not None:
        output_para_list = considered_para[:]
        output_para_list.remove(split_para)

        tree[para_names[split_para]] = {}
        for k in list(output_data_dict.keys()):
            tree[para_names[split_para]][k] = split_data_set(output_data_dict[k], output_para_list, para_names)
    return tree


# data_set_x = [['1','1','yes'], ['1','1','yes'], ['1','0','no'], ['0','1','no'], ['0','1','no']]
# para_names = ['no surfacing', 'flippers']
# considered_para = [0, 1]
data_set = np.loadtxt('lenses.txt', delimiter= '	',dtype=str)
para_names = ['age', 'prescript', 'astigmatic', 'tearRate']
considered_para = [0, 1, 2, 3]

split_tree = split_data_set(data_set, considered_para, para_names)
print(split_tree )


