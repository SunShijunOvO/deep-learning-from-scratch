# Chap 05. 误差反向传播法

layer 示意图
![Sigmoid](layer_pictures/sigmoid_layer.png "Sigmoid layer")
![Affine](layer_pictures/affine_layer.png "Affine layer")
![Affine_batch](layer_pictures/affine_layer_batch.png "Affine layer batch")
![Softmax_with_loss](layer_pictures/softmax_with_loss_layer.png "Softmax with loss layer")

# 5.7.1. 神经网络学习的全貌图
前提：神经网络中有合适的权重和偏置，调整权重和偏置以便你和训练数据的过程称为学习。神经网络的学习分为下面 4 个步骤。
- 步骤一、mini-batch<br>
    从训练数据中随机选取一部分数据。
- 步骤二、calculate gradient<br>
    计算损失函数关于各个权重参数的梯度。
- 步骤三、update parameters<br>
    将权重参数沿梯度方向进行微小更新。
- 步骤四、repeat<br>
    重复上述步骤。

# epoch, batch, iteration 的比较
epoch：使用训练集的全部数据对模型进行一次完整训练，被称之为“一代训练”；<br>
batch：使用训练集中的一小部分样本对模型权重进行一次反向传播的参数更新，这一小部分样本被称为“一批数据”；<br>
iteration：使用一个 batch 数据对模型进行一次参数更新的过程，被称之为“一次训练”。