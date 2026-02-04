# MF-for-CTR

矩阵分解做CTR任务。

## 数据集描述

1. 评分数据集，如movielens(评分1-5)，将评分转换成正负样本
2. 正负样本数据集，自带正负样本
3. 仅正样本数据集，需在训练中采样负样本以维持模型的性能。

## 模型描述

基本参数:$e_u, e_i \in \mathbb{R}^d$, $b_u, b_i \in \mathbb{R}$.

预测函数:
$$ \hat{r}_{ui} = \vec{e}_u + \vec{e}_i + \vec{b}_u + \vec{b}_i + b$$

损失函数：
$$\mathcal{L}=||r_{ui} - \hat{r}_{ui}||_2^2+\lambda(||\vec{e}_u||_2^2+||\vec{e}_i||_2^2+||\vec{b}_u||_2^2+||\vec{b}_i||_2^2)$$

## 评估函数

在验证集上有

均方误差(MSE):
$$\text{MSE} = \frac{1}{N} \sum ||r_{ui} - \hat{r}_{ui} ||$$


准确率(ACC)
$$\text{ACC} = \frac{1}{N} \sum \mathbb{I}(|r_{ui} - \hat{r}_{ui}|<0.5)$$