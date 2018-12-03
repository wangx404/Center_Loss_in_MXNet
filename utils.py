# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.patheffects as PathEffects
import mxnet as mx
from mxnet import gluon, image, nd
import gluonbook as gb
import os


def plot_features(features, labels, num_classes, fpath='features.png'):
    '''
    Plot feature map.
    绘制特征分布图。
    :param features: feature matrix 特征矩阵
    :param labels: input image label 输入图像标签
    :param num_classes: class number 数据集类别数目
    :param fpath: file path 图像储存路径
    :return None:
    '''
    name_dict = dict()
    for i in range(num_classes):
        name_dict[i] = str(i)
    
    f = plt.figure(figsize=(16, 12))
    palette = np.array(sns.color_palette("hls", num_classes))
    
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(features[:, 0], features[:, 1], lw=0, s=40,
                    c=palette[labels.astype(np.int)])
    ax.axis('off')
    ax.axis('tight')
    
    # We add the labels for each digit.
    txts = []
    for i in range(num_classes):
        # Position of each label.
        xtext, ytext = np.median(features[labels == i, :], axis=0)
        txt = ax.text(xtext, ytext, name_dict[i])
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)
    
    f.savefig(fpath)
    plt.close()


def evaluate_accuracy(data_iterator, net, ctx):
    '''
    Evaluate model accuracy on data iterator.
    评估函数的准确率并返回用于绘制特征分布的特征数据。
    :param data_iterator: data iterator 数据迭代器
    :param net: model 模型
    :param ctx: ctx
    :return acc.get()[1: accuracy rate 准确率
    :return features: 2d features of input images 输入图片2维特征
    :return predicts: predicts of input images 输入图片预测值
    :return labels: labels of input images 输入图片标签值
    '''
    acc = mx.metric.Accuracy()

    features, predicts, labels = [], [], []
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx[0])
        label = label.as_in_context(ctx[0])
        # forward compute
        output, feature = net(data)
        prediction = nd.argmax(output, axis=1)
        acc.update(preds=prediction, labels=label)
        # 存储计算的feature，output和label
        features.extend(feature.asnumpy())
        predicts.extend(prediction.asnumpy())
        labels.extend(label.asnumpy())
    # transform list to array
    features = np.array(features)
    predicts = np.array(predicts)
    labels = np.array(labels)

    return acc.get()[1], features, predicts, labels


def transform(data, label):
    '''
    Function, to float32, normalization, transpose
    数据类型变换，归一化处理，通道转换。
    '''
    return nd.transpose(data.astype(np.float32), (2, 0, 1)) / 255, label.astype(np.float32)


def data_loader(batch_size):
    '''
    Load MNIST dataset iterator.
    加载MNIST数据集迭代器。
    :param batch_size: batch size 批次大小
    :return train_iter: train dataset iterator 训练集迭代器
    :return test_iter: test dataset iterator 测试机迭代器
    '''
    train_iter = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=True, transform=transform),
                                          batch_size, shuffle=True)
    test_iter = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False, transform=transform),
                                         batch_size, shuffle=False)
    return train_iter, test_iter
