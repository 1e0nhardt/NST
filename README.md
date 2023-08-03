# NST
各种风格迁移算法

## 实验计划
- 研究input_range
    - 选定神经网络的特征层的激活数值范围，激活值的熵，激活特征的可视化，反向传播时的梯度。
    - Gram矩阵的数值范围，熵，梯度
- 研究对特征进行平滑的效果 (将风格迁移视为知识蒸馏)
    - softmax 真在平滑吗?
    - scale
    - conv
- 只用浅层特征

## 实验笔记
Inception模型和VGG模型都会受到input_range的影响。但方向相反，VGG是input_range越大，效果越好，Inception模型是input_range越小，效果越好。

Mixup

FS per channel

Code book/image size

SNeRF+NNST

