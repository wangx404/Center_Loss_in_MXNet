# Center_Loss_in_MXNet
使用MXNet Gluon实现的Center Loss的另一种方法。

## 2019.03.28更新

1. 对CenterLoss进行了略微修改（在center loss计算时，通过hist平衡类别间的损失）以保证其功能和ShownX的保持一致。
2. 在对模型的准确率进行评估时，默认使用model输出的class prediction进行。在本代码中，额外增加了使用KNN（基于欧氏距离）search进行类别预测和评估。
3. 增加了一种额外的训练方式，即仅仅使用center lossx进行训练。

## 结果讨论

1. 使用不同的损失函数进行训练后模型在验证集上的准确率如下：

| loss / evaluation method | class prediction | KNN search |
| ---- | :--:| ---- |
| softmax | 0.9855 | - |
| softmax + center loss | 0.9873 | 0.9823 |
| center loss | 0.9872 | 0.9851 |

P.S. 训练参数：epoch=30, lr=0.1, lr_step=10, lr_factor=0.1, batch_size=128, wd=1E-4, lmbd=1, alpha=0.5

## 背景

本项目的实现源于ShownX实现的[mxnet-center-loss](https://github.com/ShownX/mxnet-center-loss). 然而在阅读代码的过程中，发现Center Loss的实现并不直观。所以我就在思考能否使用其他的方式实现类似的计算过程。在阅读NLP相关的代码时意识到可以使用`gluon.nn.Embedding`代替之前代码中的`Parameter dict`。因为前者也能够实现label到feature的转换，而且输入label得到feature的过程是自动化的。

本项目中的代码大部分来源于[这里](https://github.com/ShownX/mxnet-center-loss).除了修改了原项目中对于center loss的代码之外，其他部分增加了新的注释以便理解，修改了一小部分代码以方便后续的使用。

为了能够理解本项目中的实现和之前有什么不同，你可以查看`center_loss.py`中的原始代码。简单来说，通过一个嵌入层可以直接从label得到对应的feature，而不需要调用pick函数。这样修改后除了看起来pipeline更加清楚之外，Center Loss也能够像一个常规的模型一样被使用。

## 安装依赖
```
pip install -r requirements.txt
```

## 训练
1. 使用softmax loss训练
```
$ python main.py --train --prefix=softmax-loss --epochs 30 --lr 0.1 --lr_step 10 --batch_size 128 --num_classes 10 --feature_size 2 --use_gpu True --plotting
```

2. 使用softmax + center loss训练
```
$ python main.py --train --center_loss --prefix=center-loss --lmbd 1 --alpha 0.5 --epochs 30 --lr 0.1 --lr_step 10 --batch_size 128 --num_classes 10 --feature_size 2 --use_gpu True --plotting
```

3. 在2的基础上使用center loss单独进行训练
```
$ #python main.py --second_train --prefix=center-loss --lmbd 1 --alpha 0.5  --epochs 30 --lr 0.1 --lr_step 10 --batch_size 128 --num_classes 10 --feature_size 2 --use_gpu True --plotting
```
**P.S.** 在`main.py`中你可以查看更多的训练/测试选项。例如说你可以调整batch size的大小，更改训练的epoch和学习率等等。想要得到原始论文中类似的特征分布图，则需要在训练或者测试的命令行中增加`--plotting`选项。

## 测试
1. 使用class prediction进行评估测试
```
$ python main.py --test --prefix=center-loss --lmbd 1 --batch_size 128 --num_classes 10 --feature_size 2 --use_gpu True --eval_method softmax
```

2. 使用softmax + center loss测试
```
$ python main.py --test --prefix=center-loss --lmbd 1 --batch_size 128 --num_classes 10 --feature_size 2 --use_gpu True --eval_method knn
```

**P.S.** 使用softmax训练的模型无法使用knn进行评估，因为没有特征矩阵可用。

# Center_Loss_in_MXNet
Another kind implementation of center loss using MXNet Gluon.

## Background

This project is inspired by [an implementation of center loss using MXNet](https://github.com/ShownX/mxnet-center-loss). However, when reading the implementation code, I found it was a little ugly. So I wonder if there is another kind of implementatiuon which is clean and simple. So I try to use `gluon.nn.Embedding` to replace `Parameter dict` in that origin responsitory. The experiment results showed there is no difference. 

Most codes in this responsitory comes from [here](https://github.com/ShownX/mxnet-center-loss). In addtion to my center loss class, I chaned some codes and added some annotations to make it more easier to understand.

To understand the difference of center loss implemetation between mine and my reference, you can read codes in `center_loss.py`.

## Requirements
```
pip install -r requirements.txt
```

## Training
1. Train with original softmax
```
$ python main.py --train --prefix=softmax
```

2. Train with softmax + center loss
```
$ python main.py --train --center_loss --prefix=center-loss
```

## Test
1. Test with original softmax
```
$ python main.py --test --prefix=softmax
```

2. Test with softmax + center loss
```
$ python main.py --test --prefix=center-loss
```

**P.S.** In the script `main.py`, you can find more train/test options. For example, you can change batch size, learning rate, whether to gpu for training/testing. If you want to get similar feature map in the origin paper, you can add `--plotting` in command line.
