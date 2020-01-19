import torch
from torch.utils.data import DataLoader,Dataset
import random
import numpy as np
from torch.utils.data.sampler import Sampler


#从数据集中选取support_set 和query_set
class get_SQ_set(object):

    def __init__(self,data_classes,number_class,support_sample_num,query_sample_num):
        self.data_classes=data_classes
        self.num_class=number_class
        self.support_shot_num=support_sample_num
        self.query_shot_num=query_sample_num

        classes_name = [i for i in self.data_classes.keys()]

        # temp = dict()
        self.support_data = []
        self.query_data = []
        self.support_labels=[]
        self.query_labels=[]

        for c in classes_name:
            index = [c for c in range(len(self.data_classes[c]))]
            support_index = index[:self.support_shot_num]#取第一个
            query_index = index#取全部（需要打乱？）

            self.support_data.extend(self.data_classes[c][support_index])
            self.support_labels.append(c)
            self.query_data.extend(self.data_classes[c][query_index])
            self.query_labels.extend([c for i in range(len(query_index))])

class select_query(object):

    def __init__(self,task,episode,TEST_NUMBER_WAY,QUERY_SAMPLE_NUMBER):
        self.episode = episode
        self.TEST_NUMBER_WAY = TEST_NUMBER_WAY
        self.query_shot_num = QUERY_SAMPLE_NUMBER

        self.piece_query_data = task.query_data[self.episode * self.TEST_NUMBER_WAY : (self.episode + 1) * self.TEST_NUMBER_WAY]
        self.piece_query_labels = task.query_labels[self.episode * self.TEST_NUMBER_WAY : (self.episode + 1) * self.TEST_NUMBER_WAY]

class FewShotDataset(Dataset):

    def __init__(self, task, split='sample'):
        self.task = task
        self.split = split
        self.data_roots = self.task.support_data if self.split == 'sample' else self.task.piece_query_data
        self.labels = self.task.support_labels if self.split == 'sample' else self.task.piece_query_labels

    def __len__(self):
        return len(self.image_roots)

    def __getitem__(self, idx):
        raise NotImplementedError("This is an abstract class. Subclass this class for your particular dataset.")

class Hsi_Dataset(FewShotDataset):

    def __init__(self, *args, **kwargs):
        super(Hsi_Dataset, self).__init__(*args, **kwargs)

    def __getitem__(self, idx):
        data = self.data_roots[idx]
        label = self.labels[idx]

        return data, label

#进行类的平衡
class ClassBalancedSampler(Sampler):
    ''' Samples 'num_inst' examples each from 'num_cl' pools
        of examples of size 'num_per_class' '''
    def __init__(self, num_per_class, num_cl, num_inst,shuffle=False):
        self.num_per_class = num_per_class
        self.num_cl = num_cl
        self.num_inst = num_inst
        self.shuffle = shuffle

    def __iter__(self):
        # return a single list of indices, assuming that items will be grouped by class
        # if self.shuffle:
        #     batch = [[i+j*self.num_inst for i in torch.randperm(self.num_inst)[:self.num_per_class]] for j in range(self.num_cl)]
        # else:
        #     batch = [[i+j*self.num_inst for i in range(self.num_inst)[:self.num_per_class]] for j in range(self.num_cl)]
        # batch = [item for sublist in batch for item in sublist]
        batch=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
        if self.shuffle:
            random.shuffle(batch)
        return iter(batch)

    def __len__(self):
        return 1

#获取dataset和dataloader
def get_data_loader(task, Number_way,num_per_class=1, split='train',shuffle=False):

    dataset = Hsi_Dataset(task,split=split)

    if split == 'sample':#ClassBalancedSampler 数据平衡选择器
        sampler = ClassBalancedSampler(num_per_class, Number_way, task.support_shot_num,shuffle=shuffle)
    else:
        sampler = ClassBalancedSampler(num_per_class, Number_way, task.query_shot_num,shuffle=shuffle)
    loader = DataLoader(dataset, batch_size=num_per_class*Number_way, sampler=sampler)

    return loader



