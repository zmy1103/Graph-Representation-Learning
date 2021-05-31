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

## Metric and Performance

- Mean Rank

  对于测试集的每个三元组，以预测tail实体为例，我们将**（h,r,t）**中的t用知识图谱中的每个实体来代替，然后通过`distance(h, r, t)`函数来计算距离，这样我们可以得到一系列的距离，之后按照升序将这些分数排列。

  `distance(h, r, t)`函数值是越小越好，那么在上个排列中，排的越前越好。

  看每个三元组中正确答案也就是真实的t到底能在上述序列中排多少位，比如说t1排100，t2排200，t3排60.......，之后对这些排名求平均，mean rank就得到了。

- Hit@10

  还是按照上述进行函数值排列，然后去看每个三元组正确答案是否排在序列的前十，如果在的话就计数+1

  最终 排在前十的个数/总个数 就是Hit@10

- 

## Multi-step Path

