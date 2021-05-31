# Graph-Representation-Learning

5.24-5.31:利用FB15K-237数据集，跑各个模型的XXX任务

6.1-6.7:设计场景在医疗知识图谱上实践多步路径

- 在公开数据集上复现论文中基于多步路径的模型
- 设计我们知识图谱上的场景
- 利用我们在Neo4j的数据XXX

[TOC]

## Dataset and Task

数据集FB15k-237是Freebase的子集，包含237种关系和14k种实体。训练集中包含271,115条三元组，验证集中包含17,535条三元组，测试集中包含20,466条三元组。

sota：https://paperswithcode.com/sota/link-prediction-on-fb15k-237

利用训练集进行各个模型建模，通过训练为每个实体和关系建立起向量映射，并在测试集中计算MeanRank和Hit10指标进行结果检验。

## TransE

- Principle：TransE将起始实体，关系，指向实体映射成同一空间的向量，如果（head,relation,tail）存在，那么h+r≈t

- Objective function：
  $$
  f_{r}(h, t)=\left\|\mathbf{h}_{r}+\mathbf{r}-\mathbf{t}_{r}\right\|_{2}^{2}
  $$

- Recurrence process：

  ![未命名文件.jpg](http://ww1.sinaimg.cn/large/005IQUPRgy1gr1n47kgdfj31520im77a.jpg)

- Code analysis：

  

- Result：

  经过transE建模后，在测试集的13584个实体，961个关系的 59071个三元组中，测试结果如下：

  mean rank: 
  hit@3: 
  hit@10: 

  一方面可以看出训练后的结果是有效的，但不是十分优秀，可能与transE模型的局限性有关，transE只能处理一对一的关系，不适合一对多/多对一关系。

## KBGAT

### GCN

假设图有n个节点，每个节点含有N个特征，$x = \{x_1,....x_n\}$,边关系通过邻接矩阵A来表示。图卷积操作可以分为两个步骤：

- 为了得到节点的高维表示，通过参数W做线性转换$g = Wx$
- 根据节点的邻居信息得到节点处理后的特征，需要对邻接点进行聚合$x^{'}=\sigma(\sum\alpha_{ij}g_j)$



### KBAT(Knowledge Base Attention Network)

在知识图谱的图基础上，根据relation关系，通过self-attention机制考虑邻接点实体的影响。

TODO...



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

