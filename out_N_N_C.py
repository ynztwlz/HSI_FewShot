import scipy.io as sio
from sklearn import preprocessing
import numpy as np
import pickle
from sklearn.decomposition import PCA
import random
global DATASET

# PATCH_LENGTH = 1

def index_assignment(index, row, col, pad_length):
    new_assign = {}
    for counter, value in enumerate(index):
        assign_0 = value // col + pad_length
        assign_1 = value % col + pad_length
        new_assign[counter] = [assign_0, assign_1]
    return new_assign


def assignment_index(assign_0, assign_1, col):
    new_index = assign_0 * col + assign_1
    return new_index


def select_patch(matrix, pos_row, pos_col, ex_len):
    selected_rows = matrix[range(pos_row - ex_len, pos_row + ex_len + 1)]
    selected_patch = selected_rows[:, range(pos_col - ex_len, pos_col + ex_len + 1)]
    return selected_patch


def select_small_cubic(data_size, data_indices, whole_data, patch_length, padded_data, dimension):
    small_cubic_data = np.zeros((data_size, 2 * patch_length + 1, 2 * patch_length + 1, dimension))
    data_assign = index_assignment(data_indices, whole_data.shape[0], whole_data.shape[1], patch_length)
    for i in range(len(data_assign)):
        small_cubic_data[i] = select_patch(padded_data, data_assign[i][0], data_assign[i][1], patch_length)
    return small_cubic_data

def generate_dataset(PATCH_LENGTH):
    DATASET = 'pu'
    print('Dataset:',DATASET)
    if DATASET == 'pu':
        data_hsi = sio.loadmat('./../datasets/PaviaU.mat')['paviaU']
        gt_hsi = sio.loadmat('./../datasets/PaviaU_gt.mat')['paviaU_gt']
    elif DATASET == 'in':
        data_hsi = sio.loadmat('./../datasets/Indian_pines_corrected.mat')['indian_pines_corrected']
        gt_hsi = sio.loadmat('./../datasets/Indian_pines_gt.mat')['indian_pines_gt']
    elif DATASET == 'ksc':
        data_hsi = sio.loadmat('./../datasets/KSC.mat')['ksc']
        gt_hsi = sio.loadmat('./../datasets/KSC_gt.mat')['ksc_gt']
    print('Data_Hsi_Shape:', data_hsi.shape)
    data = data_hsi.reshape(np.prod(data_hsi.shape[:2]), np.prod(data_hsi.shape[2:]))
    data = preprocessing.scale(data)
    pca = PCA(n_components=100)
    data = pca.fit_transform(data)
    gt = gt_hsi.reshape(np.prod(gt_hsi.shape[:2]), )
    nb_classes = max(gt)
    print('The class numbers of the HSI data is:', nb_classes)
    data_ = data.reshape(data_hsi.shape[0], data_hsi.shape[1], data.shape[1])
    print(data_.shape)
    padded_data = np.lib.pad(data_, ((PATCH_LENGTH, PATCH_LENGTH), (PATCH_LENGTH, PATCH_LENGTH), (0, 0)), 'constant',
                             constant_values=0)
    data_hsi = data_
    class_count = 0
    data_classes = {}
    for i in range(1, nb_classes + 1):
        index = np.where(gt == i)
        index = index[0]
        data_classes[class_count] = select_small_cubic(len(index), index, data_hsi, PATCH_LENGTH, padded_data,
                                                       data_hsi.shape[2])
        class_count += 1


    for k,v in data_classes.items():
        print(k, v.shape)

    f = open('./datasets/' + DATASET +'_data_{}_{}_{}.pickle'.format(PATCH_LENGTH*2+1, PATCH_LENGTH*2+1,data_hsi.shape[2]), 'wb')
    pickle.dump(data_classes, f)
    f.close()

if __name__== "__main__":
    generate_dataset(PATCH_LENGTH=4)