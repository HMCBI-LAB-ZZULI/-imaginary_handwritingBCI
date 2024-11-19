# 二级分类器的尝试，最终并未使用#

import os

import numpy as np
import scipy.io
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras

# 加载数据
# point this towards the top level dataset directory
rootDir = '../handwritingBCIData/'

# evaluate the RNN on these datasets
dataDirs = ['***']

# use this train/test partition
cvPart = 'HeldOutTrials'

# point this towards the specific RNN we want to evaluate
rnnOutputDir = cvPart
datas = []
labels = []
# this prevents tensorflow from taking over more than one gpu on a multi-gpu machine
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

for dataDir in dataDirs:
    dataPath = rootDir + 'RNNTrainingSteps/special_character/group1/' + dataDir + 'special_character.mat'
    data = scipy.io.loadmat(dataPath)['segments'].squeeze()  # 加载数据文件
    label = scipy.io.loadmat(dataPath)['labels'].squeeze()  # 加载标签文件
    for x, y in zip(data, label):
        if x.shape != (0, 0):  # 检查数据是否为空
            datas.append(x)
            labels.append(y)

datas = np.array(datas)
labels = np.array(labels)
print("数据总数:", len(datas))
print("标签总数:", len(labels))

# 将数据划分为训练集和测试集
train_data, test_data, train_labels, test_labels = train_test_split(datas, labels, test_size=0.2, random_state=42)


# 构建Siamese网络模型
input_shape = (None, 192)
embedding_dim = 64

# 创建Siamese网络的共享模型
shared_model = keras.Sequential([
    keras.layers.LSTM(64, input_shape=(None, 192)),
    keras.layers.Dense(128, activation='relu', input_shape=input_shape),
    keras.layers.Dense(embedding_dim, activation='relu')
])

# 输入对
input_a = keras.Input(shape=input_shape)
input_b = keras.Input(shape=input_shape)

# 嵌入向量
embeddings_a = shared_model(input_a)
embeddings_b = shared_model(input_b)

# 计算欧氏距离
distance = tf.reduce_sum(tf.square(embeddings_a - embeddings_b), axis=-1)
outputs = keras.activations.sigmoid(distance)

# 创建整个Siamese网络模型
siamese_model = keras.Model(inputs=[input_a, input_b], outputs=outputs)

# 编译模型
siamese_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 训练模型
siamese_model.fit([train_data[:, 0], train_data[:, 1]], train_labels, epochs=10, batch_size=32)

# 保存模型
siamese_model.save(rootDir + 'RNNTrainingSteps/special_character/group1/siamese_model.h5')
