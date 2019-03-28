# -*- coding: utf-8 -*-
"""
This script is used to run experiment on MNIST using softmax and center loss.
本脚本用于在MNIST数据集上使用交叉熵损失函数和中心损失函数进行试验。
"""

import argparse
import os
import time
import logging

import mxnet as mx
from mxnet import gluon
from mxnet import nd, autograd

from utils import plot_features, evaluate_accuracy, data_loader
from center_loss import CenterLoss
from lenet import LeNetPlus


def args_parser():
    """
    Command line arguments parsing function.
    命令行参数解析函数。
    """
    parser = argparse.ArgumentParser("Convolutional Neural Networks Using Center Loss")
    # File related
    parser.add_argument("--prefix", default="softmax-loss", type=str, help="prefix of LeNet++ params file") # LeNet++模型参数文件名前缀
    parser.add_argument("--ckpt_dir", default="ckpt", type=str, help="check point directory") # 检查点文件夹（储存模型参数文件）
    # Training related
    parser.add_argument("--train", action="store_true", help="train") # 训练模式
    parser.add_argument("--use_gpu", default=False, type=bool, help="whether to use gpu or not") # 是否使用gpu
    parser.add_argument("--epochs", default=10, type=int, help="epochs") # 训练轮次
    parser.add_argument("--lr", default=0.1, type=float, help="learning rate") # 初始学习率
    parser.add_argument("--lr_step", default=10, type=int, help="learning rate decay cell") # 学习率衰减周期
    parser.add_argument("--lr_factor", default=0.1, type=float, help="learning rate decay factor") # 学习率衰减因子
    parser.add_argument("--wd", default=0.0001, type=float, help="weight decay")
    parser.add_argument("--lmbd", default=0.01, type=float, help="lambda in the paper") # lambda: center loss的权重因子
    parser.add_argument("--alpha", default=0.1, type=float, help="alpha in the paper") # alpha: center loss的更新速率，即center loss的初始学习率
    parser.add_argument("--batch_size", default=128, type=int, help="batch size") # 批次大小
    parser.add_argument("--num_classes", default=10, type=int, help="number of classes") # 类别数目
    parser.add_argument("--feature_size", default=2, type=int, help="feature dimension") # 特征维度
    parser.add_argument("--center_loss", action="store_true", help="whether or not to train using center loss") # 是否使用center loss
    parser.add_argument("--plotting", action="store_true", help="whether or not to draw feature map") # 是否绘制类别特征图
    # second train
    parser.add_argument("--second_train", action="store_true", help="choose to train a model with only center loss") # 是否在第一次的基础上仅使用center loss进行训练
    # Test related
    parser.add_argument("--test", action="store_true", help="choose to test a model") # 对模型进行测试
    parser.add_argument("--out_dir", default="output", type=str, help="output directory") # 输出文件夹
    parser.add_argument("--eval_method", default="softmax", type=str, help="softmax/knn, choose one method to evaluate model's accuracy") # 模型准确率评估方法
    # 在使用softmax进行训练时，只能使用softmax评估；使用了CenterLoss之后，可选用knn进行评估（目前来看，knn评估的结果会略低于softmax）
    args = parser.parse_args()
    return args    


def train():
    """
    train model using softmax loss or softmax loss/center loss.
    训练模型。
    """
    print("Start to train...")
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    
    train_iter, test_iter = data_loader(args.batch_size)
    ctx = mx.gpu() if args.use_gpu else mx.cpu()
    
    # main model (LeNetPlus), loss, trainer
    model = LeNetPlus(classes=args.num_classes, feature_size=args.feature_size)
    model.hybridize()
    model.initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    network_trainer = gluon.Trainer(model.collect_params(),
                            optimizer="sgd", optimizer_params={"learning_rate": args.lr, "wd": args.wd}) #  "momentum": 0.9, 
    # center loss network and trainer
    if args.center_loss:
        center_loss = CenterLoss(num_classes=args.num_classes, feature_size=args.feature_size, lmbd=args.lmbd, ctx=ctx)
        center_loss.initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx) # 包含了一个center矩阵，因此需要进行初始化
        center_trainer = gluon.Trainer(center_loss.collect_params(),
                                       optimizer="sgd", optimizer_params={"learning_rate": args.alpha})
    else:
        center_loss, center_trainer = None, None
    
    smoothing_constant, moving_loss = .01, 0.0
    best_acc = 0.0
    for epoch in range(args.epochs):
        # using learning rate decay during training process
        if (epoch > 0) and (epoch % args.lr_step == 0):
            network_trainer.set_learning_rate(network_trainer.learning_rate*args.lr_factor)
            if args.center_loss:
                center_trainer.set_learning_rate(center_trainer.learning_rate*args.lr_factor)
        
        start_time = time.time()
        for i, (data, label) in enumerate(train_iter):
            data = data.as_in_context(ctx) 
            label = label.as_in_context(ctx)
            with autograd.record():
                output, features = model(data)
                loss_softmax = softmax_cross_entropy(output, label)
                # cumpute loss according to user"s choice
                if args.center_loss:
                    loss_center = center_loss(features, label)
                    loss = loss_softmax + loss_center
                else:
                    loss = loss_softmax
            
            # update 更新参数
            loss.backward() 
            network_trainer.step(args.batch_size)
            if args.center_loss:
                center_trainer.step(args.batch_size)
            
            # calculate smoothed loss value 平滑损失
            curr_loss = nd.mean(loss).asscalar()
            moving_loss = (curr_loss if ((i == 0) and (epoch == 0))
                           else (1 - smoothing_constant) * moving_loss + smoothing_constant * curr_loss) # 累计加权函数
        
        # training cost time 训练耗时
        elapsed_time = time.time() - start_time
        train_accuracy, train_ft, _, train_lb = evaluate_accuracy(train_iter, model, center_loss, args.eval_method, ctx)
        test_accuracy, test_ft, _, test_lb = evaluate_accuracy(test_iter, model, center_loss, args.eval_method, ctx)
        
        # draw feature map 绘制特征图像
        if args.plotting:
            plot_features(train_ft, train_lb, num_classes=args.num_classes,
                          fpath=os.path.join(args.out_dir, "%s-train-epoch-%d.png" % (args.prefix, epoch)))
            plot_features(test_ft, test_lb, num_classes=args.num_classes,
                          fpath=os.path.join(args.out_dir, "%s-test-epoch-%d.png" % (args.prefix, epoch)))

        logging.warning("Epoch [%d]: Loss=%f, Train-Acc=%f, Test-Acc=%f, Epoch-time=%f" % 
                        (epoch, moving_loss, train_accuracy, test_accuracy, elapsed_time))
        
        # save model parameters with the highest accuracy 保存accuracy最高的model参数
        if test_accuracy > best_acc:
            best_acc = test_accuracy
            model.save_parameters(os.path.join(args.ckpt_dir, args.prefix + "-best.params"))
            # 因为CenterLoss继承自gluon.HyperBlock,所以具有普通模型相关的对象可供调用，即可使用save_parameters/load_parameters进行参数的保存和加载。
            # 如果CenterLoss没有直接父类，那么就需要通过CenterLoss.embedding.weight.data/set_data进行数据的保存和加载。
            center_loss.save_parameters(os.path.join(args.ckpt_dir, args.prefix + "-feature_matrix.params"))


def second_train():
    """
    Train a model using only center loss based on pretrained model.
    In order to avoid feature matrix becoming zero matrix, fix CenterLoss' parameters not to train it.
    基于之前训练的模型，仅使用center loss对模型进行训练。
    为了避免在训练的过程，CenterLoss中的特征矩阵变为0矩阵，将其参数固定不对其进行训练。
    """
    print("Start to train LeNet++ with CenterLoss...")

    train_iter, test_iter = data_loader(args.batch_size)
    ctx = mx.gpu() if args.use_gpu else mx.cpu()
    
    # main model (LeNetPlus), loss, trainer
    model = LeNetPlus(classes=args.num_classes, feature_size=args.feature_size)
    model.load_parameters(os.path.join(args.ckpt_dir, args.prefix + "-best.params"), ctx=ctx, allow_missing=True)
    network_trainer = gluon.Trainer(model.collect_params(),
                            optimizer="sgd", optimizer_params={"learning_rate": args.lr, "wd": args.wd})

    center_loss = CenterLoss(num_classes=args.num_classes, feature_size=args.feature_size, lmbd=args.lmbd, ctx=ctx)
    center_loss.load_parameters(os.path.join(args.ckpt_dir, args.prefix + "-feature_matrix.params"), ctx=ctx)
    center_loss.params.setattr("grad_req", "null") # 

    smoothing_constant, moving_loss = .01, 0.0
    best_acc = 0.0
    for epoch in range(args.epochs):
        # using learning rate decay during training process
        if (epoch > 0) and (epoch % args.lr_step == 0):
            network_trainer.set_learning_rate(network_trainer.learning_rate*args.lr_factor)
        
        start_time = time.time()
        for i, (data, label) in enumerate(train_iter):
            data = data.as_in_context(ctx) 
            label = label.as_in_context(ctx)
            # only calculate loss with center loss
            with autograd.record():
                output, features = model(data)
                loss = center_loss(features, label)

            loss.backward() 
            # only update parameters of LeNet++
            network_trainer.step(args.batch_size, ignore_stale_grad=True) # without ignore_stale_grad=True, it will raise warning information
            # 去除ignore_stale_grad=True后，模型训练会出错，梯度无法进行更新
            
            # calculate smoothed loss value
            curr_loss = nd.mean(loss).asscalar()
            moving_loss = (curr_loss if ((i == 0) and (epoch == 0))
                           else (1 - smoothing_constant) * moving_loss + smoothing_constant * curr_loss)
        
        # training cost time
        elapsed_time = time.time() - start_time
        train_accuracy, train_ft, _, train_lb = evaluate_accuracy(train_iter, model, center_loss, args.eval_method, ctx)
        test_accuracy, test_ft, _, test_lb = evaluate_accuracy(test_iter, model, center_loss, args.eval_method, ctx)
        
        # draw feature map with different prefix, to make it convenient to compare features
        if args.plotting:
            plot_features(train_ft, train_lb, num_classes=args.num_classes,
                          fpath=os.path.join(args.out_dir, "%s-second-train-epoch-%d.png" % (args.prefix, epoch)))
            plot_features(test_ft, test_lb, num_classes=args.num_classes,
                          fpath=os.path.join(args.out_dir, "%s-second-test-epoch-%d.png" % (args.prefix, epoch)))

        logging.warning("Epoch [%d]: Loss=%f, Train-Acc=%f, Test-Acc=%f, Epoch-time=%f" % 
                        (epoch, moving_loss, train_accuracy, test_accuracy, elapsed_time))
        
        if test_accuracy > best_acc:
            best_acc = test_accuracy
            model.save_parameters(os.path.join(args.ckpt_dir, args.prefix + "-second-best.params"))


def test():
    """
    Test model accuracy on test dataset.
    测试模型在测试集上的准确率。
    """
    print("Start to test...")
    ctx = mx.gpu() if args.use_gpu else mx.cpu()
    
    _, test_iter = data_loader(args.batch_size)
    
    model = LeNetPlus()
    model.load_parameters(os.path.join(args.ckpt_dir, args.prefix + "-best.params"), ctx=ctx, allow_missing=True)
    # 
    center_net = CenterLoss(num_classes=args.num_classes, feature_size=args.feature_size, lmbd=args.lmbd, ctx=mx.cpu())
    center_net.load_parameters(os.path.join(args.ckpt_dir, args.prefix + "-feature_matrix.params"), ctx=ctx)
    
    start_time = time.time()
    test_accuracy, features, predictions, labels = evaluate_accuracy(test_iter, model, center_net, args.eval_method, ctx)
    elapsed_time = time.time() - start_time
    print("Test_acc: %s, Elapsed_time: %f s" % (test_accuracy, elapsed_time))
    
    # make directory
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    # draw feature map
    if args.plotting:
        plot_features(features, labels, num_classes=args.num_classes,
                      fpath=os.path.join(args.out_dir, "%s.png" % args.prefix))


if __name__ == "__main__":
    args = args_parser()
    if args.train:
        train()
    if args.second_train:
        second_train()
    if args.test:
        test()
    
