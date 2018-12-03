# -*- coding: utf-8 -*-
"""
This script is used to run experiment on MNIST using softmax and center loss.
本脚本用于在MNIST数据集上使用交叉熵损失函数和中心损失函数进行试验。
"""
import argparse
import os
import timeit
import logging

import mxnet as mx
from mxnet import gluon
from mxnet import nd, autograd

from utils import plot_features, evaluate_accuracy, data_loader
from center_loss import CenterLoss
from lenet import LeNetPlus


def args_parser():
    '''
    Command line arguments parsing function.
    命令行参数解析函数。
    '''
    parser = argparse.ArgumentParser('Convolutional Neural Networks')
    # File related
    parser.add_argument('--prefix', default='softmax-loss', type=str, help='prefix') # prefix of model parameters file
    parser.add_argument('--ckpt_dir', default='ckpt', type=str, help='check point directory')
    # Training related
    parser.add_argument('--train', action='store_true', help='train') # choose to train a model
    parser.add_argument('--use_pgu', default=False, type=bool, help='whether to use gpu or not')
    parser.add_argument('--epochs', default=10, type=int, help='epochs') # training epochs
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--lr_step', default=10, type=int, help='learning rate decay cell')
    parser.add_argument('--lr_factor', default=0.3, type=float, help='learning rate decay factor')
    parser.add_argument('--wd', default=0.0001, type=float, help='weight decay')
    parser.add_argument('--lmbd', default=0.001, type=float, help='lambda in the paper') # lambda: center loss的比重
    parser.add_argument('--alpha', default=0.1, type=float, help='alpha in the paper') # alpha: 控制center loss的更新速度
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--num_classes', default=10, type=int, help='number of classes')
    parser.add_argument('--feature_size', default=2, type=int, help='feature dimension')
    parser.add_argument('--center_loss', action='store_true', help='wgether to train using center loss')
    parser.add_argument('--plotting', action='store_true', help='whether to draw feature map') 
    # Test related
    parser.add_argument('--test', action='store_true', help='choose to test a model')
    parser.add_argument('--out_dir', default='output', type=str, help='output directory')
    args = parser.parse_args()
    return args    


def train():
    print('Start to train...')
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',')] if args.gpus != '-1' else mx.cpu()
    
    print('Loading the data...')
    train_iter, test_iter = data_loader(args.batch_size)
    # main model (LeNetPlus), loss, trainer
    model = LeNetPlus(classes=args.num_classes, feature_size=args.feature_size)
    model.hybridize()
    model.initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(model.collect_params(),
                            optimizer='sgd', optimizer_params={'learning_rate': args.lr, 'wd': args.wd}) #  'momentum': 0.9, 
    # center loss network and trainer
    if args.center_loss:
        center_loss = CenterLoss(args.num_classes, feature_size=args.feature_size, lmbd=args.lmbd)
        center_loss.initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx) # 包含了一个center矩阵，因此需要进行初始化
        trainer_center = gluon.Trainer(center_loss.collect_params(),
                                       optimizer='sgd', optimizer_params={'learning_rate': args.alpha})
    else:
        center_loss, trainer_center = None, None
    
    smoothing_constant, moving_loss = .01, 0.0
    best_acc = 0
    for epoch in range(args.epochs):
        # using learning rate decay during training process
        if (epoch > 0) and (epoch % args.lr_step == 0):
            trainer.set_learning_rate(trainer.learning_rate*args.lr_factor)
            if args.center_loss:
                trainer_center.set_learning_rate(trainer_center.learning_rate*args.lr_factor)
        
        start_time = timeit.default_timer()
        for i, (data, label) in enumerate(train_iter):
            data = data.as_in_context(ctx) 
            label = label.as_in_context(ctx)
            with autograd.record():
                output, features = model(data)
                loss_softmax = softmax_cross_entropy(output, label)
                # cumpute loss according to user's choice
                if args.center_loss:
                    loss_center = center_loss(features, label) # 加入center loss
                    loss = loss_softmax + loss_center
                else:
                    loss = loss_softmax
            # update 更新参数
            loss.backward() 
            trainer.step(data.shape[0])
            if args.center_loss:
                trainer_center.step(data.shape[0])
            # calculate smoothed loss value 平滑损失
            curr_loss = nd.mean(loss).asscalar()
            moving_loss = (curr_loss if ((i == 0) and (epoch == 0))
                           else (1 - smoothing_constant) * moving_loss + smoothing_constant * curr_loss) # 累计加权函数
        # training cost time 训练耗时
        elapsed_time = timeit.default_timer() - start_time
        train_accuracy, train_ft, _, train_lb = evaluate_accuracy(train_iter, model, ctx)
        test_accuracy, test_ft, _, test_lb = evaluate_accuracy(test_iter, model, ctx)
        # draw feature map 绘制特征图像
        if args.plotting:
            plot_features(train_ft, train_lb, num_classes=args.num_classes,
                          fpath=os.path.join(args.out_dir, '%s-train-epoch-%s.png' % (args.prefix, epoch)))
            plot_features(test_ft, test_lb, num_classes=args.num_classes,
                          fpath=os.path.join(args.out_dir, '%s-test-epoch-%s.png' % (args.prefix, epoch)))

        logging.warning("Epoch [%d]: Loss=%f, Train-Acc=%f, Test-Acc=%f, Epoch-time=%f" % 
                        (epoch, moving_loss, train_accuracy, test_accuracy, elapsed_time))
        # save model parameters with the highest accuracy 保存accuracy最高的model参数
        if test_accuracy > best_acc:
            best_acc = test_accuracy
            model.save_parameters(os.path.join(args.ckpt_dir, args.prefix + '-best.params'))


def test():
    '''
    Test model accuracy on test dataset.
    测试模型在测试集上的准确率。
    '''
    print('Start to test...')
    ctx = mx.gpu() if args.use_gpu else mx.cpu()
    
    _, test_iter = data_loader(args.batch_size)
    
    model = LeNetPlus()
    model.load_parameters(os.path.join(args.ckpt_dir, args.prefix + '-best.params'), ctx=ctx, allow_missing=True)
    
    start_time = timeit.default_timer()
    test_accuracy, features, predictions, labels = evaluate_accuracy(test_iter, model, ctx)
    elapsed_time = timeit.default_timer() - start_time
    
    print("Test_acc: %s, Elapsed_time: %f s" % (test_accuracy, elapsed_time))
    # make directory
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    # draw feature map
    if args.plotting:
        plot_features(features, labels, num_classes=args.num_classes,
                      fpath=os.path.join(args.out_dir, '%s.png' % args.prefix))


if __name__ == '__main__':
    args = args_parser()
    if args.train:
        train()
    if args.test:
        test()
    
