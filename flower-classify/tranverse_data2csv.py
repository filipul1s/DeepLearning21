# -*- coding:utf-8 -*-
# @Date :2021/7/26 11:41
# @Author:KittyLess
# @name: tranverse_data2csv
import pandas as pd
import os


# 1 给train数据集打上标签 并 创建字典表示 保存csv文件到train_data_csv
def tranverse_label2num(train_file):
    num = len(train_file)
    return dict(zip(train_file,range(num)))

def tranverse_images(path,is_Train=True):
    if(is_Train):
        labels = pd.DataFrame()
        train_data_root = [file for file in os.listdir(os.path.join(path,'train'))]
        class2num = tranverse_label2num(train_data_root)
        for item in train_data_root:
            flower = [image for image in os.listdir(os.path.join(path,'train',item))]
            num = len(flower)
            for index in range(num):
                flower[index] = item +'/' + flower[index]
            labels_data = pd.DataFrame({'flower':flower,'lables':item,'labels_num':class2num[item]})
            labels = pd.concat((labels,labels_data))
        labels.head()
        labels.to_csv('../data/flower_data/train_data_csv.csv')
        return '--***生成train.csv成功***--'
    else:
        labels = pd.DataFrame()
        test_data_root = [file for file in os.listdir(os.path.join(path, 'val'))]
        for item in test_data_root:
            flower = [image for image in os.listdir(os.path.join(path, 'val', item))]
            num = len(flower)
            for index in range(num):
                flower[index] = item + '/' + flower[index]
            labels_data = pd.DataFrame({'flower': flower})
            labels = pd.concat((labels, labels_data))
        labels.head()
        labels.to_csv('../data/flower_data/test_data_csv.csv')
        return '--***生成test.csv成功***--'



