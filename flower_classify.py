# -*- coding:utf-8 -*-
# @Date :2021/7/26 1:07
# @Author:KittyLess
# @name: flower_classify

import os
import torch
import torch.utils.data
import torch.nn as nn
import pandas as pd
import numpy as np
from d2l import torch as d2l
import torchvision
import tranverse_data2csv
from tqdm import tqdm
from PIL import Image



def tranverse_label2num(train_file):
    num = len(train_file)
    return dict(zip(train_file,range(num)))

data_iter = '../data/flower_data/'
lables_dataframe = pd.read_csv(os.path.join(data_iter,'train_data_csv.csv'))

flower_labels = sorted(list(set(lables_dataframe['lables'])))  # set删掉重复的 sort排序
n_classes = len(flower_labels)  # 类别长度
# 把label转成对应的数字
class_to_num = dict(zip(flower_labels, range(n_classes)))
# 再转换回来，方便最后预测的时候使用
num_to_class = {v: k for k, v in class_to_num.items()}

#get_train_csv = tranverse_data2csv.tranverse_images(os.path.join(os.getcwd(),'../data/flower_data'),is_Train=True)
#get_test_csv = tranverse_data2csv.tranverse_images(os.path.join(os.getcwd(),'../data/flower_data'),is_Train=False)

# 2、数据增强
train_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.RandomVerticalFlip(p=0.5),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

test_augs = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224,224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# 3、读取数据
def read_flower_data(mode='train'):
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../data/flower_data"))  # get data root path
    data_root_child = 'train' if mode == 'train' else 'val'
    if mode == 'train':
        dataset_csv = pd.read_csv(os.path.join(data_root,'train_data_csv.csv'),header=None)
        aug = train_augs
        label_arr = np.asarray(dataset_csv.iloc[1:,1])
    else:
        dataset_csv = pd.read_csv(os.path.join(data_root,'test_data_csv.csv'),header=None)
        aug = test_augs
    image_arr = np.asarray(dataset_csv.iloc[1:,0])
    # 计算length这里会把表头记录进来 所以要减1
    dataset_csv = dataset_csv.iloc[1:]
    train_len = len(dataset_csv.index) - 1
    all_images,targets = [],[]
    for index,target in dataset_csv.iterrows():

        img_as_img = Image.open(os.path.join(data_root,data_root_child,target[0]))
        img_as_img = aug(img_as_img)
        all_images.append(img_as_img)
        #all_images.append(
        #    torchvision.io.read_image(os.path.join(data_root,data_root_child,target[0]))
        #)
        targets.append(list(target))
    return all_images,targets

#_,targets = read_flower_data()
#print(targets)

class FlowerDataset(torch.utils.data.Dataset):
    def __init__(self,mode='train'):
        self.mode = mode
        self.features,self.labels = read_flower_data(mode)
        print('read ' + str(len(self.features)) + (
            f' training/valid examples' if mode == 'train' or mode == 'valid' else f' test examples'))
    def __getitem__(self, index):
        if self.mode == 'val':
            return self.features[index].float()
        else:
            lable = self.labels[index][1]
            label = class_to_num[lable]
            return (self.features[index].float(),label)

    def __len__(self):
        return len(self.features)

batch_size = 16
def load_data_flowers(batch_size):
    train_iter = torch.utils.data.DataLoader(FlowerDataset(mode='train'),batch_size,shuffle=True)
    test_iter = torch.utils.data.DataLoader(FlowerDataset(mode='val'),batch_size)
    return train_iter,test_iter

train_iter,test_iter = load_data_flowers(batch_size)

# 展示图片
#onebatch = next(iter(train_iter))
#imgs = (onebatch[0][0:10].permute(0,2,3,1)) # 在做图片增强的时候已经做了标准化
#axes = d2l.show_images(imgs,2,5,titles=onebatch[1][0:10],scale=2)
#d2l.plt.show()



# 4、定义网络模型
'''
net = nn.Sequential(nn.Conv2d(3,48,kernel_size=(11,11),stride=(4,4),padding=1),nn.ReLU(),
    nn.MaxPool2d(kernel_size=3,stride=2),
    nn.Conv2d(48,128,kernel_size=(5,5),padding=2),nn.ReLU(),
    nn.MaxPool2d(kernel_size=3,stride=2),
    nn.Conv2d(128,192,kernel_size=(3,3),padding=1),nn.ReLU(),
    nn.Conv2d(192,192,kernel_size=(3,3),padding=1),nn.ReLU(),
    nn.Conv2d(192,128,kernel_size=(3,3),padding=1),nn.ReLU(),
    nn.MaxPool2d(kernel_size=3,stride=2),nn.Flatten(),
    nn.Linear(128 * 6 * 6,2048),nn.ReLU(),nn.Dropout(p=0.5),
    nn.Linear(4096,2048),nn.ReLU(),nn.Dropout(p=0.5),
    nn.Linear(2048,5))

def __init__weight(m):
    if isinstance(m,nn.Conv2d):
        nn.init.xavier_normal_(m.weight)

net.apply(__init__weight)
'''

def get_devce():
    return  'cuda' if torch.cuda.is_available() else "cpu"

device = get_devce()

net = torchvision.models.resnet18(pretrained=True)
net.fc = nn.Linear(net.fc.in_features,5)
nn.init.xavier_normal_(net.fc.weight)
nn.device = device


net.to(device)
loss_f = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(),lr=0.002)
epochs = 10
best_acc = 0.0
train_steps = len(train_iter)

def train_batch_ch13(net, X, y, loss, trainer, devices):
    """Train for a minibatch with mutiple GPUs (defined in Chapter 13)."""
    if isinstance(X, list):
        # Required for BERT fine-tuning (to be covered later)
        X = [x.to(devices) for x in X]
    else:
        X = X.to(devices)
    y = y.to(devices)
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = d2l.accuracy(pred, y)
    return train_loss_sum, train_acc_sum

def train_batch(net):
    net.to(get_devce())
    for epoch in range(epochs):
        train_loss = []
        train_accs = []
        metric = d2l.Accumulator(4)
        running_loss = 0.0
        train_bar = tqdm(train_iter)
        for step,data in enumerate(train_bar):
            images,labels = data
            l, acc = train_batch_ch13(net, images, labels, loss_f, optimizer, device)

            metric.add(l,acc,labels.shape[0],labels.numel())
            train_accs.append(acc)

        #训练集的平均损失和精度是记录值的平均值。
        train_loss = metric[0] / len(train_accs)
        print('acc:', metric[1], 'labels.numerl():', metric[3], 'acc_lens:', len(train_accs), 'labels.shape[0]:', metric[2])
        train_acc = metric[1] / metric[3]

        # 打印信息
        print(f"[ Train | {epoch + 1:03d}/{epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

    print('Finished Training')

# test ------  #
def train_and_pred(net):
    net.eval()
    predictions = []
    for i,X in enumerate(test_iter):
        with torch.no_grad():
            pred = net(X.to(device))

        # 以最大的pred类为预测 这里不是很懂 code meaning
        predictions.extend(pred.argmax(dim=-1).cpu().numpy().tolist())
    preds = []
    for i in predictions:
        preds.append(num_to_class[i])
    test_data = pd.read_csv(os.path.join(data_iter, 'test_data_csv.csv'))
    test_data['label'] = pd.Series(preds)  # 将预测的类型名整理成一位数组
    submission = pd.concat([test_data['flower'], test_data['label']], axis=1)
    submission.to_csv('submission.csv', index=False)

train_batch(net)

train_and_pred(net)

# 展示图片
onebatch = next(iter(test_iter))
read_result = pd.read_csv(os.path.join(data_iter, 'test_data_csv.csv'))
print(read_result)
#titles = read_result['label'][0:10]
#imgs = (onebatch[0][0:10].permute(0,2,3,1)) # 在做图片增强的时候已经做了标准化
#axes = d2l.show_images(imgs,2,5,titles=titles,scale=2)
#d2l.plt.show()