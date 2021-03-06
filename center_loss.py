from mxnet import gluon, nd
import numpy as np

# This class is defined by ShownX (https://github.com/ShownX)
# 本类由ShownX定义
class _CenterLoss(gluon.HybridBlock):
    """
    Center Loss: A Discriminative Feature Learning Approach for Deep Face Recognition
    """
    def __init__(self, num_classes, feature_size, lmbd, **kwargs):
        '''
        :param num_classes: class number, 类别数目
        :param feature_size: output feature size, 特征大小
        :param lmbd: lmbd coefficient to control center loss, lmbd系数 用于控制center loss的大小
        '''
        super(CenterLoss, self).__init__(**kwargs)
        self._num_classes = num_classes
        self._feature_size = feature_size
        self._lmda = lmbd
        self.centers = self.params.get('centers', shape=(num_classes, feature_size)) # 获取参数矩阵

    def hybrid_forward(self, F, feature, label, centers):
        '''
        :param feature: output feature matrix, mini-batch的输出特征矩阵
        :param label: input label matrix, mini-batch的输入标签
        :param centers: features of class, 类别特征矩阵
        '''
        # 计算label的统计分布 hist=（label0:counter0, label1:counter1）
        hist = F.array(np.bincount(label.asnumpy().astype(int))) 
        # 取出label对应的频次，用于对相应label的损失降权（所有的向量都得到同等程度的更新）
        # 在demo中此行的存在与否可能并没有太大影响，但是对于实际中使用的类别不均衡的数据集，可能会影响到优化结果
        # center_count = (num_label_0, num_label_1, )
        centers_count = F.take(hist, label) 
        # 取出label对应位置的特征向量
        centers_selected = F.take(centers, label)
        # 计算输出特征和特征矩阵的差值
        diff = feature - centers_selected 
        # lmbd×欧氏距离/统计数目
        loss = self._lmda * 0.5 * F.sum(F.square(diff), axis=1) / centers_count 
        # mean value
        return F.mean(loss, axis=0, exclude=True)


# This class is define by myself. As you can see, it's much simpler.
# In this class, I use Embedding layer to get label related class feature.
# Besides, I ignore the decentralization mechanism.
# 我定义的center loss类，一个更加简洁、更易理解的定义。
# 在本类中，我使用了MXNet中的嵌入层完成label到特征向量的转换，并进行center loss的计算。
# 除此之外，通过将count注释掉即可忽略参数更新过程中的降权机制。
# 在center loss中定义的center loss似乎是不必要的，因为即便在每次的更新中，每个类别均获得类似的更新速度
# 但是由于在整体的数据集中数据不均衡存在，那么类别获得更新的几率仍然是不同的。
class CenterLoss(gluon.HybridBlock):
    def __init__(self, num_classes, feature_size, lmbd, ctx, **kwargs):
        '''
        :param num_classes: class number, 类别数目
        :param feature_size: output feature size, 特征大小
        :param lmbd: lmbd coefficient to control center loss, lmbd系数 用于控制center loss的大小
        :param ctx: compute context, 确保向量的上下文保持一致
        '''
        super(CenterLoss, self).__init__(**kwargs)
        self.feature_size = feature_size
        self.lmbd = lmbd
        self.embedding = gluon.nn.Embedding(num_classes, feature_size)
        self.ctx = ctx
    
    def forward(self, feature, label):
        '''
        :param feature: output feature matrix, mini-batch的输出特征矩阵
        :param label: input label matrix, mini-batch的输入标签
        '''
        # embedding to get label related features 通过嵌入层得到label对应的feature
        label_count = nd.array(np.bincount(label.asnumpy().astype(int)), self.ctx)
        count = nd.take(label_count, label)
        
        embeddings= self.embedding(label)
        # reshape from (batch, 1, feature_size) to (batch, feature_size) reshape矩阵
        embeddings = embeddings.reshape((-1, self.feature_size)) 
        diff = feature - embeddings # calculate diff 计算差值
        loss = self.lmbd * 0.5 * nd.sum(nd.square(diff), axis=1) / count
        return nd.mean(loss, axis=0, exclude=True) # 这一步似乎是完全没有必要的
