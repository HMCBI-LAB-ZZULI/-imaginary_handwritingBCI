import numpy as np
import scipy.io
import os
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras import regularizers
from tensorflow.keras.utils import to_categorical

# 加载数据
rootDir = '../handwritingBCIData/'

# 评估RNN的数据集
dataDirs = ['t5.2019.05.08', 't5.2019.11.25', 't5.2019.12.09', 't5.2019.12.11', 't5.2019.12.18',
            't5.2019.12.20', 't5.2020.01.06', 't5.2020.01.08', 't5.2020.01.13', 't5.2020.01.15']

# 训练/测试分区
cvPart = 'HeldOutTrials'

# 指向特定RNN的评估目录
rnnOutputDir = cvPart

# 防止tensorflow占用多GPU机器上的多个GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

# 初始化空列表
datas = []
labels = []

# 读取数据和标签
for dataDir in dataDirs:
    dataPath = os.path.join(rootDir, 'RNNTrainingSteps/special_character/group2/', dataDir + 'special_character.mat')
    data = scipy.io.loadmat(dataPath)['segments'].squeeze()  # 加载数据文件
    label = scipy.io.loadmat(dataPath)['labels'].squeeze()  # 加载标签文件
    for x, y in zip(data, label):
        if x.shape != (0, 0):  # 检查数据是否为空
            datas.append(x)
            labels.append(y)

# 对数据进行预处理
max_sequence_length = max([len(data) for data in datas])  # 计算序列的最大长度

# 创建序列数据的张量
data_sequences = tf.keras.preprocessing.sequence.pad_sequences(datas, maxlen=max_sequence_length, padding='post', dtype='float32')

# 创建标签的张量
label_categories = np.unique(labels)
label_mapping = {label: index for index, label in enumerate(label_categories)}
label_indices = np.array([label_mapping[label] for label in labels])

# 将标签转换为one-hot编码
one_hot_labels = to_categorical(label_indices)

# 数据集划分为K折
k = 5  # 设置K的值
skf = StratifiedKFold(n_splits=k, shuffle=True)
fold = 1

for train_index, test_index in skf.split(data_sequences, labels):
    # 划分训练集和测试集
    train_data, test_data = data_sequences[train_index], data_sequences[test_index]
    train_labels, test_labels = one_hot_labels[train_index], one_hot_labels[test_index]

    # 构建更复杂的RNN模型
    model = keras.Sequential([
        keras.layers.Masking(mask_value=0, input_shape=(max_sequence_length, 192)),  # 掩码层，用于忽略填充的零值
        keras.layers.LSTM(64, return_sequences=True, kernel_regularizer=regularizers.l2(0.01)),
        keras.layers.LSTM(32, return_sequences=True, kernel_regularizer=regularizers.l2(0.01)),
        keras.layers.LSTM(32, return_sequences=False, kernel_regularizer=regularizers.l2(0.01)),
        keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        keras.layers.Dropout(0.5),  # 添加Dropout层，丢弃50%的神经元
        keras.layers.Dense(len(label_categories), activation='softmax')  # 使用softmax激活函数进行多分类
    ])

    # 设置初始学习率
    initial_learning_rate = 0.01
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps=10000, decay_rate=0.9)
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

    # 编译模型
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # 训练模型
    model.fit(train_data, train_labels, epochs=100, batch_size=32, validation_split=0.2)

    # 评估模型
    test_loss, test_acc = model.evaluate(test_data, test_labels)
    print('Fold:', fold)
    print('Test accuracy:', test_acc)

    # 保存模型
    save_path = os.path.join(rootDir, f'RNNTrainingSteps/special_character/group2/siamese_model{fold:02d}.h5')
    model.save(save_path)

    fold += 1
