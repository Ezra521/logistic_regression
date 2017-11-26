import h5py
import numpy as np


def load_dataset():
    train_dataset = h5py.File('./datasets/train_catvnoncat.h5', "r")
    """
    for key in train_dataset.keys():
        print('wqk-train_dataset-0000key')
        print(key)
        print('wqk-train_dataset-1111name')
        print(train_dataset[key].name)
        print('wqk-train_dataset-2222shape')
        print(train_dataset[key].shape)
        print('wqk-train_dataset-3333value')
        print(train_dataset[key].value)
    """
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # 你的训练数据的特征
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # 你的训练数据的标签

    test_dataset = h5py.File('./datasets/test_catvnoncat.h5', "r")

    """
    print('\n')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('\n')
    for key in test_dataset.keys():
        print('wqk-test_dataset-0000key')
        print(key)
        print('wqk-test_dataset-1111name')
        print(test_dataset[key].name)
        print('wqk-test_dataset-2222shape')
        print(test_dataset[key].shape)
        print('wqk-test_dataset-3333value')
        print(test_dataset[key].value)
    """
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # 你的测试数据的特征
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # 你的测试数据的标签

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
load_dataset()
# train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
# print('test_set_y:' + str(test_set_y))
# print(test_set_x_orig)
#
# print(test_set_x_orig.shape)
