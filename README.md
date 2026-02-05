# MF-for-CTR

矩阵分解方法来进行点击预测任务。

论文:[MATRIX FACTORIZATION TECHNIQUES FOR RECOMMENDER SYSTEMS](https://datajobs.com/data-science-repo/Recommender-Systems-%5BNetflix%5D.pdf)

实现该论文的方法，运用在点击预测(Click Through Rate, CTR)任务上。该论文方法的目的是预测电影评分，如1-5之间。该项目是探索该方法在点击预测方法上的效果。

## 任务及数据集描述

任务：点击预测任务，给定一个用户ID和物品ID，预测用户是否会点击该物品。

数据集描述：
1. 评分数据集，如movielens(评分1-5)。需要转换：将评分转换成点击正负样本。
2. 点击正负样本数据集，自带点击正负样本。
3. 仅点击正样本数据集。需要转换：在训练过程中从未交互数据采样样本，当做负样本，以维持模型的性能。

### MF模型描述

基本参数: $\vec{e}_u, \vec{e}_i \in \mathbb{R}^d$, $b_u, b_i \in \mathbb{R}$.

点击预测函数(是否点击):

$$ \hat{r}_{ui} = \sigma(\vec{e}_u + \vec{e}_i + \vec{b}_u + \vec{b}_i + b),$$

损失函数：

$$\mathcal{L}=\text{CE}(r_{ui}, \hat{r}_{ui})+\lambda(||\vec{e}_u||_2^2+||\vec{e}_i||_2^2+||\vec{b}_u||_2^2+||\vec{b}_i||_2^2).$$

### 评估函数

对模型性能进行评估有：

交叉熵(CE):

$$ \text{CE} = \frac{1}{N} \sum $$

点击预测准确率(ACC):

$$\text{ACC} = \frac{1}{N} \sum \mathbb{I}(|r_{ui} - \hat{r}_{ui}|<0.5).$$
