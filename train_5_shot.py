import torch
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import math
import numpy as np
import utils.read_data as rd
import pickle

# Hyper Parameters参数初始化
FEATURE_DIM = 64
RELATION_DIM = 8
TEST_EPISODE = 1000
TEST_NUMBER_WAY = 9
NUMBER_WAY = 20
SUPPORT_SAMPLE_NUMBER = 5
QUERY_SAMPLE_NUMBER = 19
LEARNING_RATE = 0.001
EPISODE = 1000000
GPU = 0


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
    # 载入高光谱数据
    f = open('./datasets/train_classes.pickle', 'rb')
    data_classes = pickle.load(f)
    f.close()

    #初始化网络
    feature_encoder = CNNEncoder()
    relation_network = RelationNetwork(FEATURE_DIM,RELATION_DIM)

    feature_encoder.apply(weights_init)
    relation_network.apply(weights_init)

    feature_encoder.cuda(GPU)
    relation_network.cuda(GPU)

    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(),lr=LEARNING_RATE)
    feature_encoder_scheduler = StepLR(feature_encoder_optim,step_size=100000,gamma=0.5)
    relation_network_optim = torch.optim.Adam(relation_network.parameters(),lr=LEARNING_RATE)
    relation_network_scheduler = StepLR(relation_network_optim,step_size=100000,gamma=0.5)

    #载入模型
    if os.path.exists(
            str("./models/hsi_feature_encoder_" + str(NUMBER_WAY) + "way_" + str(SUPPORT_SAMPLE_NUMBER) + "shot.pkl")):
        feature_encoder.load_state_dict(torch.load(
            str("./models/hsi_feature_encoder_" + str(NUMBER_WAY) + "way_" + str(SUPPORT_SAMPLE_NUMBER) + "shot.pkl"),
            map_location='cuda:0'))
        print("load feature encoder success")
    if os.path.exists(
            str("./models/hsi_relation_network_" + str(NUMBER_WAY) + "way_" + str(SUPPORT_SAMPLE_NUMBER) + "shot.pkl")):
        relation_network.load_state_dict(torch.load(
            str("./models/hsi_relation_network_" + str(NUMBER_WAY) + "way_" + str(SUPPORT_SAMPLE_NUMBER) + "shot.pkl"),
            map_location='cuda:0'))
        print("load relation network success")

    # Step 3: build graph
    print("Training...")

    last_accuracy = 0.0
    for episode in range(EPISODE):
        feature_encoder_scheduler.step(episode)
        relation_network_scheduler.step(episode)

        # 在随机选取的类中每一类随机选取a（here is 1）个样本index，全部集合起来作为support_set
        # 在随机选取的类中每一类随机选取b(here is 19)个样本index,全部集合起来作为query_set
        task = rd.get_SQ_set(data_classes,NUMBER_WAY,SUPPORT_SAMPLE_NUMBER,QUERY_SAMPLE_NUMBER)

        #获取dataset和dataloader
        sample_dataloader = rd.get_data_loader(task,NUMBER_WAY,num_per_class=SUPPORT_SAMPLE_NUMBER,split="support")
        query_dataloader = rd.get_data_loader(task,NUMBER_WAY,num_per_class=QUERY_SAMPLE_NUMBER,split="query")

        # 取出每一个样本数据
        supports,support_labels = sample_dataloader.__iter__().next()
        querys,querys_labels = query_dataloader.__iter__().next()


        # querys_labels=querys_labels.astype('long')
        #改变成10*10
        supports = supports.unsqueeze(1)
        supports = supports.reshape(supports.shape[0], supports.shape[1], 10, 10)
        querys = querys.unsqueeze(1)
        querys = querys.reshape(querys.shape[0], querys.shape[1], 10, 10)
        supports = supports.type(torch.FloatTensor)
        querys = querys.type(torch.FloatTensor)
        # 送入网络 calculate features
        # sample和query同时输入feature_encoder
        support_features = feature_encoder(Variable(supports).cuda(GPU))
        support_features = support_features.view(NUMBER_WAY,SUPPORT_SAMPLE_NUMBER,FEATURE_DIM,support_features.shape[-2],support_features.shape[-1])
        support_features = torch.sum(support_features,1).squeeze(1)
        query_features = feature_encoder(Variable(querys).cuda(GPU))
        support_features_ext = support_features.unsqueeze(0).repeat(QUERY_SAMPLE_NUMBER*NUMBER_WAY,1,1,1,1)
        query_features_ext = query_features.unsqueeze(0).repeat(NUMBER_WAY,1,1,1,1)
        query_features_ext = torch.transpose(query_features_ext,0,1)
        relation_pairs = torch.cat((support_features_ext,query_features_ext),2).view(-1,FEATURE_DIM*2,query_features_ext.shape[-2],support_features_ext.shape[-1])
        relations = relation_network(relation_pairs).view(-1,NUMBER_WAY)

        mse = nn.MSELoss().cuda(GPU)
        one_hot_labels = Variable(torch.zeros(QUERY_SAMPLE_NUMBER * NUMBER_WAY, NUMBER_WAY).scatter_(1, querys_labels.view(-1,1).long(), 1)).cuda(GPU)
        loss = mse(relations, one_hot_labels)

        # training
        feature_encoder.zero_grad()
        relation_network.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(feature_encoder.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(relation_network.parameters(), 0.5)
        feature_encoder_optim.step()
        relation_network_optim.step()

        # 计算精度
        total_rewards=0
        _, predict_labels = torch.max(relations.data, 1)
        predict_labels = predict_labels.cpu().numpy()
        querys_labels = querys_labels.cpu().numpy()
        rewards = [1 if predict_labels[j] == querys_labels[j] else 0 for j in range(NUMBER_WAY * QUERY_SAMPLE_NUMBER)]
        total_rewards += np.sum(rewards)
        accuracy = total_rewards/(NUMBER_WAY * QUERY_SAMPLE_NUMBER)

        if (episode+1)%100 == 0:
            print("episode:",episode+1,"loss",loss.item(),"accuracy:",accuracy*100,'%')

        if (episode + 1) % 5000 == 0:
            print('Testing:')
            # 载入测试高光谱数据
            f = open('./datasets/test_classes.pickle', 'rb')
            test_data_classes = pickle.load(f)
            f.close()
            total_rewards = 0
            for i in range(TEST_EPISODE):
                task = rd.get_SQ_set(test_data_classes, TEST_NUMBER_WAY, SUPPORT_SAMPLE_NUMBER, QUERY_SAMPLE_NUMBER)
                test_sample_dataloader = rd.get_data_loader(task, TEST_NUMBER_WAY, num_per_class=SUPPORT_SAMPLE_NUMBER,
                                                            split="support")
                test_query_dataloader = rd.get_data_loader(task, TEST_NUMBER_WAY, num_per_class=SUPPORT_SAMPLE_NUMBER,
                                                           split="query")
                test_supports, test_support_labels = test_sample_dataloader.__iter__().next()
                test_querys, test_querys_labels = test_query_dataloader.__iter__().next()

                test_supports = test_supports.unsqueeze(1)
                test_supports = test_supports.reshape(test_supports.shape[0], test_supports.shape[1], 10, 10)
                test_querys = test_querys.unsqueeze(1)
                test_querys = test_querys.reshape(test_querys.shape[0], test_querys.shape[1], 10, 10)
                test_supports = test_supports.type(torch.FloatTensor)
                test_querys = test_querys.type(torch.FloatTensor)
                #print(test_supports.shape)
                test_support_features = feature_encoder(Variable(test_supports).cuda(GPU))
                test_support_features = test_support_features.view(TEST_NUMBER_WAY, SUPPORT_SAMPLE_NUMBER, FEATURE_DIM,
                                                                   test_support_features.shape[-2], test_support_features.shape[-1])
                test_support_features = torch.sum(test_support_features, 1).squeeze(1)
                test_query_features = feature_encoder(Variable(test_querys).cuda(GPU))

                test_support_features_ext = test_support_features.unsqueeze(0).repeat(
                    SUPPORT_SAMPLE_NUMBER * TEST_NUMBER_WAY, 1, 1, 1, 1)
                test_query_features_ext = test_query_features.unsqueeze(0).repeat(TEST_NUMBER_WAY, 1, 1, 1, 1)
                test_query_features_ext = torch.transpose(test_query_features_ext, 0, 1)
                test_relation_pairs = torch.cat((test_support_features_ext, test_query_features_ext), 2).view(-1,
                                                                                                              FEATURE_DIM * 2,
                                                                                                              test_query_features_ext.shape[-2], test_query_features_ext.shape[-1])
                test_relations = relation_network(test_relation_pairs).view(-1, TEST_NUMBER_WAY)

                _, predict_labels = torch.max(test_relations.data, 1)
                predict_labels = predict_labels.cpu().numpy()
                rewards = [1 if predict_labels[j] == test_querys_labels[j] else 0 for j in
                           range(TEST_NUMBER_WAY * SUPPORT_SAMPLE_NUMBER)]  # QUERY_SAMPLE_NUMBER
                total_rewards += np.sum(rewards)
            test_accuracy = total_rewards / (
                        TEST_NUMBER_WAY * SUPPORT_SAMPLE_NUMBER) / TEST_EPISODE  # QUERY_SAMPLE_NUMBER
            print("test accuracy:", test_accuracy * 100, '%')
            if test_accuracy > last_accuracy:
                # save networks
                torch.save(feature_encoder.state_dict(), str(
                    "./models/hsi_feature_encoder_" + str(NUMBER_WAY) + "way_" + str(
                        SUPPORT_SAMPLE_NUMBER) + "shot.pkl"))
                torch.save(relation_network.state_dict(), str(
                    "./models/hsi_relation_network_" + str(NUMBER_WAY) + "way_" + str(
                        SUPPORT_SAMPLE_NUMBER) + "shot.pkl"))

                print("save networks for episode:", episode)

                last_accuracy = test_accuracy
if __name__ == '__main__':
    main()