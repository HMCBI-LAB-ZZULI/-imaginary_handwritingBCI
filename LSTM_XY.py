import numpy as np
import scipy.io
import os
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers, layers

# Loading data
# Point to the top-level dataset directory
rootDir = '../handwritingBCIData/'

# Evaluate RNN on these datasets
dataDirs = ['t5.2019.05.08', 't5.2019.11.25', 't5.2019.12.09', 't5.2019.12.11', 't5.2019.12.18',
            't5.2019.12.20', 't5.2020.01.06', 't5.2020.01.08', 't5.2020.01.13', 't5.2020.01.15']

# Use this train/test partition
cvPart = 'HeldOutTrials'

# Point to the specific RNN we want to evaluate
rnnOutputDir = cvPart

# Prevent tensorflow from occupying multiple GPUs on a multi-GPU machine
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

# Initialize an empty list
datas = []
labels = []

# # Reading data and tags
for dataDir in dataDirs:
    dataPath = os.path.join(rootDir, 'your path')
    data = scipy.io.loadmat(dataPath)['segments'].squeeze()  #Loading data files
    label = scipy.io.loadmat(dataPath)['labels'].squeeze()  # Loading label files
    for x, y in zip(data, label):
        if x.shape != (0, 0):  # Check if the data is empty
            datas.append(x)
            labels.append(y)

# Create a tensor of labels
label_categories = np.unique(labels)
label_mapping = {label: index for index, label in enumerate(label_categories)}
label_indices = np.array([label_mapping[label] for label in labels])

max_length = max(len(seq) for seq in datas)
datas = tf.keras.preprocessing.sequence.pad_sequences(datas, maxlen=max_length, padding='post')

# Divide the training set and test set
train_data_np, test_data_np, train_labels, test_labels = train_test_split(datas, label_indices, test_size=0.2, random_state=42)

# Build complex LSTM models and adjust them appropriately according to the size of the dataset
model = keras.Sequential([
    layers.Masking(mask_value=0.0, input_shape=(max_length, 192)),  # Mask layer to ignore zero values ​​for padding
    layers.LSTM(64, return_sequences=True, kernel_regularizer=regularizers.l2(0.01)),
    layers.LSTM(32, return_sequences=True, kernel_regularizer=regularizers.l2(0.01)),
    layers.LSTM(32, return_sequences=False, kernel_regularizer=regularizers.l2(0.01)),
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.Dropout(0.3),  # Add a Dropout layer to discard 50% of the neurons
    layers.Dense(len(label_categories), activation='sigmoid')  # Using sigmoid activation function for multi-classification
])

# Set the initial learning rate
initial_learning_rate = 0.01
lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps=10000, decay_rate=0.9)
optimizer = keras.optimizers.RMSprop(learning_rate=lr_schedule)

# Compile the model
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#Training the model
model.fit(train_data_np, train_labels, epochs=100, batch_size=32, validation_split=0.2)

# Evaluating the Model
test_loss, test_acc = model.evaluate(test_data_np, test_labels)
print('Test accuracy:', test_acc)

# Save the model
model.save(os.path.join(rootDir, 'your_path/siamese_model001.h5'))
