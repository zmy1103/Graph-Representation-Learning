# Graph-Representation-Learning

5.24-5.31:利用FB15K-237数据集，复现几个模型的预测任务

6.1-6.7:设计场景在医疗知识图谱上实践多步路径

- 在FB15K上复现论文中基于多步路径的TKFGE模型
- 设计我们知识图谱上的应用场景
- 利用我们在Neo4j导入的数据进行实验

[TOC]

## Dataset and Task

数据集FB15k-237是Freebase的子集，包含237种关系和14k种实体。训练集中包含271,115条三元组，验证集中包含17,535条三元组，测试集中包含20,466条三元组。

sota：https://paperswithcode.com/sota/link-prediction-on-fb15k-237

利用训练集进行各个模型建模，通过训练为每个实体和关系建立起向量映射，并在测试集中计算MeanRank和Hit10指标进行结果检验。

## TransE

### Principle

TransE将起始实体，关系，指向实体映射成同一空间的向量，如果（head,relation,tail）存在，那么h+r≈t

### Objective function

$$
f_{r}(h, t)=\left\|\mathbf{h}_{r}+\mathbf{r}-\mathbf{t}_{r}\right\|_{2}^{2}
$$

### Recurrence process

![未命名文件.jpg](http://ww1.sinaimg.cn/large/005IQUPRgy1gr1n47kgdfj31520im77a.jpg)

### Evaluation code analysis

![](http://ww1.sinaimg.cn/large/005IQUPRly1gr2l3dvyxpj31a80qe15o.jpg)

### Result：

经过transE建模后，在测试集的13584个实体，961个关系的 59071个三元组中，测试结果如下：

mean rank: 
hit@3: 
hit@10: 

一方面可以看出训练后的结果是有效的，但不是十分优秀，可能与transE模型的局限性有关，transE只能处理一对一的关系，不适合一对多/多对一关系。

## ConvE

### Principle

![](http://ww1.sinaimg.cn/large/005IQUPRgy1gr2nvj7n8bj31xc0ns4qp.jpg)

对三元组（h,r,t）,将（h,r）进行网络变换后，与t进行比较，使得损失最小化

### calculation process

前向传播：

- 嵌入：将h,r嵌入为向量，并转换成矩阵进行拼接
- 卷积：利用卷积核对拼接后的矩阵计算二维卷机
- 全连接：将卷机得到的特征矩阵拉平为向量，通过全连接层映射为嵌入维度的向量
- 内积：实体嵌入矩阵与全连接层得到的向量进行内积，然后进行softmax计算

损失函数：
$$
\begin{array}{l}
L(p, t)=\frac{1}{K} \sum_{k}\left(t_{k} \log \left(p_{k}\right)+\left(1-t_{k}\right) \log \left(1-p_{k}\right)\right) \\
t_{k}=\left\{\begin{array}{l}
1 & \text { 当 } e 1 \text { 与 } e k \text { 存在关系 } r \\
0 & \text { 当e1与 } e k \text { 不存在关系 } r
\end{array}\right.
\end{array}
$$
模型细节：

- 使用relu函数做非线形激活函数
- 在嵌入层、卷积层和全连接层后使用batch normalisation
- 使用dropout以减少过拟合
- 内积层相当于1-N scoring，大大提高了计算效率

### Result

ConvE Predictive Performance任务在各个数据集上的表现：

| Dataset   | MR   | MRR  | Hits@10 | Hits@3 | Hits@1 |
| --------- | ---- | ---- | ------- | ------ | ------ |
| FB15k     | 64   | 0.75 | 0.87    | 0.80   | 0.67   |
| WN18      | 504  | 0.94 | 0.96    | 0.95   | 0.94   |
| FB15k-237 | 246  | 0.32 | 0.49    | 0.35   | 0.24   |
| WN18RR    | 4766 | 0.43 | 0.51    | 0.44   | 0.39   |
| YAGO3-10  | 2792 | 0.52 | 0.66    | 0.56   | 0.45   |
| Nations   | 2    | 0.82 | 1.00    | 0.88   | 0.72   |
| UMLS      | 1    | 0.94 | 0.99    | 0.97   | 0.92   |
| Kinship   | 2    | 0.83 | 0.98    | 0.91   | 0.73   |

模型参数少计算比较快，对于高入度的图节点表现更好

## KBAT

### GCN

假设图有n个节点，每个节点含有N个特征，$x = \{x_1,....x_n\}$,边关系通过邻接矩阵A来表示。图卷积操作可以分为两个步骤：

- 为了得到节点的高维表示，通过参数W做线性转换$g = Wx$
- 根据节点的邻居信息得到节点处理后的特征，需要对邻接点进行聚合$x^{'}=\sigma(\sum\alpha_{ij}g_j)$



### KBAT(Knowledge Base Attention Network)

![image-20210531184310746](https://tva1.sinaimg.cn/large/008i3skNly1gr1sscnorpj31eo0lctfo.jpg)



Knowledge Embedding 就是希望通过模型去学习一个有效的实体和关系embedding，以及一个打分模型$f$。以下为模型主要计算流程。该模型主要是从GCN的基础上借鉴注意力机制，在知识图谱这个特定场景中构建了Knowledge Base Attention Network，并在公开数据集上不同任务中取得了不错的效果。



Relation embedding中，我们通过对邻接点加以注意力机制后进行聚合，并对实体和关系特征进行了线性变换以映射到更高维度。公式如下，其中$c_{ijk}$表示一个三元组的向量表示，$h_i,h_j,g_k$表示实体头和尾结点以及关系的向量表示。

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gr1t4d99b5j30bi02eq2y.jpg" alt="image-20210531185440493" style="zoom:50%;" />

模型结构并没有显示的使用注意力机制，而是使用一个简单的神经网络层作为注意力机制层，其计算方式如下：

另外，使用Self-attention机制，计算$b_{ijk}$，对于所有邻居节点的三元组$t_{ij}^k$，其中$h_i,h_j$表示第i,j个实体，$W_2$是线性变换矩阵。

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gr1sws9rt4j30fa03ejri.jpg" alt="image-20210531184726892" style="zoom:50%;" />

然后使用Softmax将attention的输出归一化，得到$\alpha_{ijk}$，$N_i$表示与$e_i, e_j$邻接的节点。

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gr1tbpf60qj30ig05cwey.jpg" alt="image-20210531184935359" style="zoom:50%;" />

对于每个实体$e_i$其新的embedding通过注意力机制权重聚合邻居信息：

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gr1t2fi459j30f6042aaa.jpg" alt="image-20210531185249375" style="zoom:50%;" />

然后使用与multi-head attention相似的机制，用于稳定学习过程和得到更多邻居节点的信息：

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gr1t413bahj30ec04uq37.jpg" alt="image-20210531185421335" style="zoom:50%;" />

最后再对新生成的embedding矩阵G做线性变化：

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gr1t6rvrqqj308003swef.jpg" alt="image-20210531185700150" style="zoom:50%;" />

至此通过上述流程，可以得到一个KnowLedge Base Attention Layer。

## Metric and Performance

### MRR(Mean reciprocal rank)

MRR是一个常用来衡量搜索算法效果的指标，目前被广泛用在允许返回多个结果的问题，或者目前还比较难以解决的问题中（由于如果只返回top 1的结果，准确率或召回率会很差，所以在技术不成熟的情况下，先返回多个结果）。在这类问题中，系统会对每一个返回的结果给一个置信度（打分），然后根据置信度排序，将得分高的结果排在前面返回。

MRR得核心思想很简单：返回的结果集的优劣，跟第一个正确答案的位置有关，第一个正确答案越靠前，结果越好。计算公式如下：

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gr1q3p43e5j30fa04kglu.jpg" alt="image-20210531171016941" style="zoom:50%;" />

其中$|Q|$是测试数据个数，$rank_i$是对于第i个用户，推荐列表中第一个在ground-truth结果中的item所在的排列位置。该指标越大越好，取值范围是0~1。



### MR(Mean rank)

对于每组测试数据，模型都会对所有预测结果打分，根据打分对结果排序，假如真实正确的结果位置在$i$，MR表示的就是所有测试数据对应正确结果在预测中排序位置的平均值。

比如第一，二，三组数据真实结果模型预测他们排名分别为10，20，30，$MR = (10 + 20 + 30)/ 3=20$。



### Hit ratio(Hit@1, Hit@3, Hit@10)

在推荐排序问题中，HR是一种常用的衡量召回率的指标，当然链接预测问题同样可用。$Hit@K$，表示对某组数据预测出排前K个元素中，是否存在真实的结果，如果存在则表示命中+1。最后计算命中数占测试数据的比例即为$Hit@K$的值



### Performance 

|        | MR   | MRR   | Hit@1 | Hit@3 | Hit@10 |
| ------ | ---- | ----- | ----- | ----- | ------ |
| TransE | 323  | 0.279 | 0.198 | 0.376 | 0.441  |
| KBAT   | 210  | 0.518 | 0.46  | 0.54  | 0.626  |



## Multi-step Path

