import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from sklearn.decomposition import PCA
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import pickle
import read_get_maps as rdmaps
import scipy as sp
import scipy.stats
import scipy.io as sio

# Hyper Parameters参数初始化
# FEATURE_DIM = 64
# RELATION_DIM = 8
# TEST_EPISODE = 1000
# HIDDEN_UNIT = 10
# VALIDATION_SPLIT = 0.4
# PATCH_LENGTH = 4
# img_rows = 2 * PATCH_LENGTH + 1
# img_cols = 2 * PATCH_LENGTH + 1
# NUMBER_WAY = 20
# TEST_NUMBER_WAY = 9
# SUPPORT_SAMPLE_NUMBER = 1
# QUERY_SAMPLE_NUMBER = 9
# LEARNING_RATE = 0.001
# EPISODE = 1000000
# GPU = 0
# SAMPLE_NUMBER = 200  # 每一类样本取的数量
# N_COMPONENTS = 100  # 统一光谱维度，将所有的光谱统一降到一定的维度,pca降维后的维度
FEATURE_DIM = 64
RELATION_DIM = 8
TEST_EPISODE = 1000
TEST_NUMBER_WAY = 16
NUMBER_WAY = 20
SUPPORT_SAMPLE_NUMBER = 1
QUERY_SAMPLE_NUMBER = 19
LEARNING_RATE = 0.001
GPU = 0



# def mean_confidence_interval(data, confidence=0.95):
#     a = 1.0*np.array(data)
#     n = len(a)
#     m, se = np.mean(a),scipy.stats.sem(a)
#     h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
#     return m,h

# def index_assignment(index, row, col, pad_length):
#     new_assign = {}
#     for counter, value in enumerate(index):
#         assign_0 = value // col + pad_length
#         assign_1 = value % col + pad_length
#         new_assign[counter] = [assign_0, assign_1]
#     return new_assign
#
# def assignment_index(assign_0, assign_1, col):
#     new_index = assign_0 * col + assign_1
#     return new_index
#
# def select_patch(matrix, pos_row, pos_col, ex_len):
#     selected_rows = matrix[range(pos_row - ex_len, pos_row + ex_len + 1)]
#     selected_patch = selected_rows[:, range(pos_col - ex_len, pos_col + ex_len + 1)]
#     return selected_patch
#
# def select_small_cubic(data_size, data_indices, whole_data, patch_length, padded_data, dimension):
#     small_cubic_data = np.zeros((data_size, 2 * patch_length + 1, 2 * patch_length + 1, dimension))
#     data_assign = index_assignment(data_indices, whole_data.shape[0], whole_data.shape[1], patch_length)
#     for i in range(len(data_assign)):
#         small_cubic_data[i] = select_patch(padded_data, data_assign[i][0], data_assign[i][1], patch_length)
#     return small_cubic_data

#保存图片
def classification_map(map, ground_truth, dpi, save_path):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(ground_truth.shape[1]*2.0/dpi, ground_truth.shape[0]*2.0/dpi)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)
    ax.imshow(map)
    fig.savefig(save_path, dpi=dpi)

    return 0

#list to map
def list_to_colormap(x_list):
    y = np.zeros((x_list.shape[0], 3))
    for index, item in enumerate(x_list):
        if item == 0:
            y[index] = np.array([255, 0, 0]) / 255.
        if item == 1:
            y[index] = np.array([0, 255, 0]) / 255.
        if item == 2:
            y[index] = np.array([0, 0, 255]) / 255.
        if item == 3:
            y[index] = np.array([255, 255, 0]) / 255.
        if item == 4:
            y[index] = np.array([0, 255, 255]) / 255.
        if item == 5:
            y[index] = np.array([255, 0, 255]) / 255.
        if item == 6:
            y[index] = np.array([192, 192, 192]) / 255.
        if item == 7:
            y[index] = np.array([128, 128, 128]) / 255.
        if item == 8:
            y[index] = np.array([128, 0, 0]) / 255.
        if item == 9:
            y[index] = np.array([128, 128, 0]) / 255.
        if item == 10:
            y[index] = np.array([0, 128, 0]) / 255.
        if item == 11:
            y[index] = np.array([128, 0, 128]) / 255.
        if item == 12:
            y[index] = np.array([0, 128, 128]) / 255.
        if item == 13:
            y[index] = np.array([0, 0, 128]) / 255.
        if item == 14:
            y[index] = np.array([255, 165, 0]) / 255.
        if item == 15:
            y[index] = np.array([255, 215, 0]) / 255.
        if item == 16:
            y[index] = np.array([0, 0, 0]) / 255.

    return y

# def mean_confidence_interval(data, confidence=0.95):
#     a = 1.0 * np.array(data)
#     n = len(a)
#     m, se = np.mean(a), scipy.stats.sem(a)
#     h = se * sp.stats.t._ppf((1 + confidence) / 2., n - 1)
#     return m, h
data_hsi = sio.loadmat('../../datasets/Salinas_corrected.mat')['salinas_corrected']
gt_hsi = sio.loadmat('../../datasets/Salinas_gt.mat')['salinas_gt']
data = data_hsi.reshape(np.prod(data_hsi.shape[:2]), np.prod(data_hsi.shape[2:]))
#data = preprocessing.scale(data)
pca = PCA(n_components=100)
data = pca.fit_transform(data)
gt = gt_hsi.reshape(np.prod(gt_hsi.shape[:2]), )



class CNNEncoder(nn.Module):
    """docstring for ClassName"""
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(1,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())
                        #nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())
        self.layer4 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out # 64


class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self,input_size,hidden_size):
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(128,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,1)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0),-1)
        out = torch.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        return out

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())


def main():

    # 初始化网络
    feature_encoder = CNNEncoder()
    relation_network = RelationNetwork(FEATURE_DIM, RELATION_DIM)

    # feature_encoder.apply(weights_init)
    # relation_network.apply(weights_init)

    feature_encoder.cuda(GPU)
    relation_network.cuda(GPU)

    # feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(), lr=0.001)
    # feature_encoder_scheduler = StepLR(feature_encoder_optim, step_size=100000, gamma=0.5)
    # relation_network_optim = torch.optim.Adam(relation_network.parameters(), lr=LEARNING_RATE)
    # relation_network_scheduler = StepLR(relation_network_optim, step_size=100000, gamma=0.5)

    # 载入模型
    if os.path.exists(
            str("../models/hsi_feature_encoder_" + str(NUMBER_WAY) + "way_" + str(SUPPORT_SAMPLE_NUMBER) + "shot.pkl")):
        feature_encoder.load_state_dict(torch.load(
            str("../models/hsi_feature_encoder_" + str(NUMBER_WAY) + "way_" + str(SUPPORT_SAMPLE_NUMBER) + "shot.pkl"),
            map_location='cuda:0'))
        print("load feature encoder success")
    if os.path.exists(
            str("../models/hsi_relation_network_" + str(NUMBER_WAY) + "way_" + str(SUPPORT_SAMPLE_NUMBER) + "shot.pkl")):
        relation_network.load_state_dict(torch.load(
            str("../models/hsi_relation_network_" + str(NUMBER_WAY) + "way_" + str(SUPPORT_SAMPLE_NUMBER) + "shot.pkl"),
            map_location='cuda:0'))
        print("load relation network success")

    f = open('../datasets/test_classes.pickle', 'rb')
    test_data_classes = pickle.load(f)
    f.close()
    task = rdmaps.get_SQ_set(test_data_classes, TEST_NUMBER_WAY, SUPPORT_SAMPLE_NUMBER, TEST_NUMBER_WAY)
    total_data_num=len(task.query_data)


    N_epoch = total_data_num//TEST_NUMBER_WAY#少了余数,后期补上
    total_rewards=0
    total_accuracy = 0.0
    total_predict = []
    print("Drawing...")
    for episode in range(N_epoch):
        # test

        accuracies = []
        #每次循环选固定的sample和选取test_number_way个数据,依照epoch依次选下去
        query_task=rdmaps.select_query(task, episode, TEST_NUMBER_WAY,SUPPORT_SAMPLE_NUMBER)

        sample_dataloader = rdmaps.get_data_loader(task, TEST_NUMBER_WAY, num_per_class=SUPPORT_SAMPLE_NUMBER,
                                               split="sample")
        query_dataloader = rdmaps.get_data_loader(query_task, TEST_NUMBER_WAY, num_per_class=SUPPORT_SAMPLE_NUMBER,
                                             split="query")

        test_supports, test_supports_labels = sample_dataloader.__iter__().next()
        test_querys, test_querys_labels = query_dataloader.__iter__().next()

        supports = test_supports.unsqueeze(1)
        supports = supports.reshape(supports.shape[0], supports.shape[1], 10, 10).type(torch.FloatTensor)
        querys = test_querys.unsqueeze(1)
        querys = querys.reshape(querys.shape[0], querys.shape[1], 10, 10).type(torch.FloatTensor)


        # calculate features
        supports_features = feature_encoder(Variable(supports).cuda(GPU))  # 5x64
        supports_features = supports_features.view(TEST_NUMBER_WAY, SUPPORT_SAMPLE_NUMBER, FEATURE_DIM, 5, 5)
        supports_features = torch.sum(supports_features, 1).squeeze(1)
        querys_features = feature_encoder(Variable(querys).cuda(GPU))  # 20x64

        # calculate relations
        # each batch sample link to every samples to calculate relations
        # to form a 100x128 matrix for relation network
        supports_features_ext = supports_features.unsqueeze(0).repeat(SUPPORT_SAMPLE_NUMBER * TEST_NUMBER_WAY, 1, 1,
                                                                      1, 1)
        querys_features_ext = querys_features.unsqueeze(0).repeat(TEST_NUMBER_WAY, 1, 1, 1, 1)
        querys_features_ext = torch.transpose(querys_features_ext, 0, 1)

        relation_pairs = torch.cat((supports_features_ext, querys_features_ext), 2).view(-1, FEATURE_DIM * 2, 5, 5)
        relations = relation_network(relation_pairs).view(-1, TEST_NUMBER_WAY)

        _, predict_labels = torch.max(relations.data, 1)

        predict_labels = list(predict_labels.cpu().numpy())
        total_predict.extend(predict_labels)


        rewards = [1 if predict_labels[j] == test_querys_labels[j] else 0 for j in
                   range(TEST_NUMBER_WAY * SUPPORT_SAMPLE_NUMBER)]

        total_rewards += np.sum(rewards)
        each_acc=(np.sum(rewards) / (TEST_NUMBER_WAY * SUPPORT_SAMPLE_NUMBER)) * 100
        print('accuracy:',each_acc, '%')

    print("aver_accuracy:", (total_rewards / (total_data_num-total_data_num%TEST_NUMBER_WAY))*100,'%')

    # 将图中没有标记类的点全部填充成0
    zero = list(torch.zeros(np.prod(data_hsi.shape[:2]) - len(total_predict)).numpy())
    total_predict.extend(zero)
    x = np.ravel(total_predict)
    gt = gt_hsi.flatten()



    print('-------Save the result in mat format--------')
    x_re = np.reshape(x, (gt_hsi.shape[0], gt_hsi.shape[1]))
    sio.savemat('../datasets/Predict_Salinas.mat', {'Salinas': x_re})

    y_list = list_to_colormap(x)
    y_gt = list_to_colormap(gt)

    y_re = np.reshape(y_list, (gt_hsi.shape[0], gt_hsi.shape[1], 3))
    gt_re = np.reshape(y_gt, (gt_hsi.shape[0], gt_hsi.shape[1], 3))

    classification_map(y_re, gt_hsi, 300,
                       '../datasets/' + 'Salinas' + '_' + '.png')
    classification_map(gt_re, gt_hsi, 300,
                       '../datasets/' + 'Salinas' + '_gt.png')
    print('------Get classification maps successful-------')



if __name__ == '__main__':
    main()
