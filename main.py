"""
This script is used to run experiment on MNIST using softmax and center loss
"""
import argparse
import os
import timeit
import logging

import mxnet as mx
from mxnet import gluon
from mxnet import nd, autograd

from utils import plot_features, evaluate_accuracy, data_loader
from center_loss import CenterLoss, MyCenterLoss
from models import LeNetPlus


def train():
    print('Start to train...')
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',')] if args.gpus != '-1' else mx.cpu()
    print('Loading the data...')

    train_iter, test_iter = data_loader(args.batch_size)
    # 模型
    model = LeNetPlus(classes=args.num_classes, feature_size=args.feature_size)
    model.hybridize()
    model.initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
    # loss
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    # trainer
    trainer = gluon.Trainer(model.collect_params(),
                            optimizer='sgd', optimizer_params={'learning_rate': args.lr, 'wd': args.wd}) #  'momentum': 0.9, 
    # 使用softmax函数时应该使用SGD进行训练，否则就会出现loss=nan的情况（SGD+Momentum的形式更容易出现数值溢出问题）
    # 需要一个相对较小的学习率用于训练，否则会因为训练速度过大导致loss=nan
    # center loss network and trainer
    if args.center_loss:
        #center_loss = CenterLoss(args.num_classes, feature_size=args.feature_size, lmbd=args.lmbd)
        center_loss = MyCenterLoss(args.num_classes, feature_size=args.feature_size, lmbd=args.lmbd)
        center_loss.initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx) # 包含了一个center矩阵，因此需要进行初始化
        trainer_center = gluon.Trainer(center_loss.collect_params(),
                                       optimizer='sgd', optimizer_params={'learning_rate': args.alpha})
    else:
        center_loss, trainer_center = None, None

    smoothing_constant, moving_loss = .01, 0.0

    best_acc = 0
    for epoch in range(args.epochs):
        if (epoch > 0) and (epoch % args.lr_step == 0):
            trainer.set_learning_rate(trainer.learning_rate*args.lr_factor)
            if args.center_loss:
                trainer_center.set_learning_rate(trainer_center.learning_rate*args.lr_factor)
        start_time = timeit.default_timer()

        for i, (data, label) in enumerate(train_iter):
            data = data.as_in_context(ctx[0]) # 单gpu训练
            label = label.as_in_context(ctx[0])
            with autograd.record():
                output, features = model(data)
                loss_softmax = softmax_cross_entropy(output, label)
                if args.center_loss:
                    loss_center = center_loss(features, label) # 加入center loss
                    loss = loss_softmax + loss_center
                else:
                    loss = loss_softmax
            # 更新主网络
            loss.backward() 
            trainer.step(data.shape[0])
            # 更新center 网络
            if args.center_loss:
                trainer_center.step(data.shape[0])

            curr_loss = nd.mean(loss).asscalar()
            # 使用一个epoch内的平均loss即可
            moving_loss = (curr_loss if ((i == 0) and (epoch == 0))
                           else (1 - smoothing_constant) * moving_loss + smoothing_constant * curr_loss) # 累计加权函数

        elapsed_time = timeit.default_timer() - start_time

        train_accuracy, train_ft, _, train_lb = evaluate_accuracy(train_iter, model, ctx) # 在train网络上仍然需要一个重新遍历？
        test_accuracy, test_ft, _, test_lb = evaluate_accuracy(test_iter, model, ctx)
        # 绘制图像
        if args.plotting:
            plot_features(train_ft, train_lb, num_classes=args.num_classes,
                          fpath=os.path.join(args.out_dir, '%s-train-epoch-%s.png' % (args.prefix, epoch)))
            plot_features(test_ft, test_lb, num_classes=args.num_classes,
                          fpath=os.path.join(args.out_dir, '%s-test-epoch-%s.png' % (args.prefix, epoch)))

        logging.warning("Epoch [%d]: Loss=%f, Train-Acc=%f, Test-Acc=%f, Epoch-time=%f" % 
                        (epoch, moving_loss, train_accuracy, test_accuracy, elapsed_time))
        #logging.warning("Epoch [%d]: Loss=%f" % (epoch, moving_loss))
        #logging.warning("Epoch [%d]: Train-Acc=%f" % (epoch, train_accuracy))
        #logging.warning("Epoch [%d]: Test-Acc=%f" % (epoch, test_accuracy))
        #logging.warning("Epoch [%d]: Elapsed-time=%f" % (epoch, elapsed_time))
        # 保存accuracy最高的model参数
        if test_accuracy > best_acc:
            best_acc = test_accuracy
            model.save_parameters(os.path.join(args.ckpt_dir, args.prefix + '-best.params'))


def test():
    print('Start to test...')
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',')] if args.gpus != '-1' else mx.cpu()

    _, test_iter = data_loader(args.batch_size)

    model = LeNetPlus()
    model.load_parameters(os.path.join(args.ckpt_dir, args.prefix + '-best.params'), ctx=ctx, allow_missing=True)

    start_time = timeit.default_timer()
    test_accuracy, features, predictions, labels = evaluate_accuracy(test_iter, model, ctx)
    elapsed_time = timeit.default_timer() - start_time

    print("Test_acc: %s, Elapsed_time: %f s" % (test_accuracy, elapsed_time))

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    if args.plotting:
        plot_features(features, labels, num_classes=args.num_classes,
                      fpath=os.path.join(args.out_dir, '%s.png' % args.prefix))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Convolutional Neural Networks')
    # File related
    parser.add_argument('--prefix', default='softmax', type=str, help='prefix')
    parser.add_argument('--ckpt_dir', default='ckpt', type=str, help='check point directory')
    # Training related
    parser.add_argument('--gpus', default='0', type=str, help='gpus') # -1时使用cpu训练
    parser.add_argument('--epochs', default=10, type=int, help='epochs')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--lr_step', default=20, type=int, help='learning rate decrease cell')
    parser.add_argument('--lr_factor', default=0.3, type=float, help='learning rate decrease factor')
    parser.add_argument('--lmbd', default=0.001, type=float, help='lambda in the paper') # lambda: center loss的比重
    parser.add_argument('--alpha', default=0.1, type=float, help='alpha in the paper') # alpha: 控制center loss的更新速度
    parser.add_argument('--wd', default=0.0001, type=float, help='weight decay')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--num_classes', default=10, type=int, help='number of classes')
    parser.add_argument('--feature_size', default=2, type=int, help='feature dimension')
    parser.add_argument('--train', action='store_true', help='train')
    parser.add_argument('--center_loss', action='store_true', help='train using center loss')
    parser.add_argument('--plotting', action='store_true', help='generate figure') # 是否绘图
    # Test related
    parser.add_argument('--test', action='store_true', help='test')
    parser.add_argument('--out_dir', default='my_output', type=str, help='output directory')

    args = parser.parse_args()

    if args.train:
        train()

    if args.test:
        test()
