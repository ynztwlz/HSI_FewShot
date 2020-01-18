import torch
from torch.utils.data import DataLoader,Dataset
import random
import numpy as np
from torch.utils.data.sampler import Sampler

#从数据集中选取support_set 和query_set
class get_SQ_set(object):
    def __init__(self, data_classes, number_class, support_sample_num, query_sample_num):
        self.data_classes = data_classes
        self.num_class = number_class
        self.support_shot_num = support_sample_num
        self.query_shot_num = query_sample_num

        classes_name = [i for i in self.data_classes.keys()]
        np.random.shuffle(classes_name)
        epoch_classes = classes_name[:self.num_class]

        labels = np.array(range(len(epoch_classes)))#将随机选取的类重新定义为（1-class_number）
        labels = dict(zip(epoch_classes, labels))#将类的名称和标记按照字典格式一一对应
        temp = dict()
        self.support_data = []
        self.query_data = []
        self.support_labels=[]
        self.query_labels=[]

        for c in epoch_classes:
            temp[c] = random.sample(list(data_classes[c]), len(data_classes[c]))[0:200]
            self.support_data.extend(temp[c][:self.support_shot_num])
            self.support_labels.extend(
                [labels[c] for i in range(self.support_shot_num)])  # 将标记根据字典转化成【0,1,2,3...】用于后面的one-hot编码
            self.query_data.extend(temp[c][self.support_shot_num: self.support_shot_num + self.query_shot_num])
            self.query_labels.extend([labels[c] for i in range(self.query_shot_num)])

class FewShotDataset(Dataset):

    def __init__(self, task, split='support'):
        self.task = task
        self.split = split
        self.data_roots = self.task.support_data if self.split == 'support' else self.task.query_data
        self.labels = self.task.support_labels if self.split == 'support' else self.task.query_labels

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
        if self.shuffle:
            batch = [[i+j*self.num_inst for i in torch.randperm(self.num_inst)[:self.num_per_class]] for j in range(self.num_cl)]
        else:
            batch = [[i+j*self.num_inst for i in range(self.num_inst)[:self.num_per_class]] for j in range(self.num_cl)]
        batch = [item for sublist in batch for item in sublist]

        if self.shuffle:
            random.shuffle(batch)
        return iter(batch)

    def __len__(self):
        return 1

#获取dataset和dataloader
def get_data_loader(task, Number_way,num_per_class=1, split='train',shuffle=False):

    dataset = Hsi_Dataset(task,split=split)

    if split == 'support':#ClassBalancedSampler 数据平衡选择器
        sampler = ClassBalancedSampler(num_per_class, Number_way, task.support_shot_num,shuffle=shuffle)
    else:
        sampler = ClassBalancedSampler(num_per_class, Number_way, task.query_shot_num,shuffle=shuffle)
    loader = DataLoader(dataset, batch_size=num_per_class*Number_way, sampler=sampler)

    return loader