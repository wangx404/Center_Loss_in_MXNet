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
3. Train with softmax + center loss in my own way

## Test
1. Test with original softmax
```
$ python main.py --test --prefix=softmax
```

2. Test with softmax + center loss
```
$ python main.py --test --prefix=center-loss
```

