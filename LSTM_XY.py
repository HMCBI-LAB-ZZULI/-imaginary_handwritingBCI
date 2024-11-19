import numpy as np
import scipy.io
import os
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers, layers

# 加载数据
# 指向顶级数据集目录
rootDir = '../handwritingBCIData/'

# 在这些数据集上评估RNN
dataDirs = ['t5.2019.05.08', 't5.2019.11.25', 't5.2019.12.09', 't5.2019.12.11', 't5.2019.12.18',
            't5.2019.12.20', 't5.2020.01.06', 't5.2020.01.08', 't5.2020.01.13', 't5.2020.01.15']

# 使用这个训练/测试分区
cvPart = 'HeldOutTrials'

# 指向我们要评估的特定RNN
rnnOutputDir = cvPart

# 防止tensorflow占用多GPU机器上的多个GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

# 初始化空列表
datas = []
labels = []

# 读取数据和标签
for dataDir in dataDirs:
    dataPath = os.path.join(rootDir, 'your path')
    data = scipy.io.loadmat(dataPath)['segments'].squeeze()  # 加载数据文件
    label = scipy.io.loadmat(dataPath)['labels'].squeeze()  # 加载标签文件
    for x, y in zip(data, label):
        if x.shape != (0, 0):  # 检查数据是否为空
            datas.append(x)
            labels.append(y)

# 创建标签的张量
label_categories = np.unique(labels)
label_mapping = {label: index for index, label in enumerate(label_categories)}
label_indices = np.array([label_mapping[label] for label in labels])

max_length = max(len(seq) for seq in datas)
datas = tf.keras.preprocessing.sequence.pad_sequences(datas, maxlen=max_length, padding='post')

# 划分训练集和测试集
train_data_np, test_data_np, train_labels, test_labels = train_test_split(datas, label_indices, test_size=0.2, random_state=42)

# 构建复杂的LSTM模型,根据数据集大小适当调整
model = keras.Sequential([
    layers.Masking(mask_value=0.0, input_shape=(max_length, 192)),  # 掩码层，用于忽略填充的零值
    layers.LSTM(64, return_sequences=True, kernel_regularizer=regularizers.l2(0.01)),
    layers.LSTM(32, return_sequences=True, kernel_regularizer=regularizers.l2(0.01)),
    layers.LSTM(32, return_sequences=False, kernel_regularizer=regularizers.l2(0.01)),
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.Dropout(0.3),  # 添加Dropout层，丢弃50%的神经元
    layers.Dense(len(label_categories), activation='sigmoid')  # 使用sigmoid激活函数进行多分类
])

# 设置初始学习率
initial_learning_rate = 0.01
lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps=10000, decay_rate=0.9)
optimizer = keras.optimizers.RMSprop(learning_rate=lr_schedule)

# 编译模型
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(train_data_np, train_labels, epochs=100, batch_size=32, validation_split=0.2)

# 评估模型
test_loss, test_acc = model.evaluate(test_data_np, test_labels)
print('Test accuracy:', test_acc)

# 保存模型
model.save(os.path.join(rootDir, 'RNNTrainingSteps/special_character/group1/siamese_model001.h5'))
