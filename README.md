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

## KBGAT

## Metric and Performance

## Multi-step Path

