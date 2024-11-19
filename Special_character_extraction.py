import tensorflow as tf
import numpy as np
import scipy.io
import os
import warnings

# 初始化设置
def init_setup():
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    warnings.simplefilter(action='ignore', category=FutureWarning)

# 加载和处理数据
def process_data(rootDir, dataDirs, cvPart, target_characters, group_number):
    rnnOutputDir = cvPart
    inferenceSaveDir = os.path.join(rootDir, 'RNNTrainingSteps', 'YOUR_SPCA_INPUT_RNNInference', rnnOutputDir)
    group_path = os.path.join(rootDir, 'RNNTrainingSteps', 'special_character', f'group{group_number}')
    os.makedirs(group_path, exist_ok=True)

    for dataset in dataDirs:
        print(f'Processing dataset: {dataset}')
        specdat = {'segments': [], 'labels': []}
        sentenceDat = scipy.io.loadmat(os.path.join(rootDir, 'Datasets', dataset, 'sentences.mat'))
        sentences = sentenceDat['sentencePrompt']
        # 加载要提取的数据集和对应标签
        eeg_data = scipy.io.loadmat(os.path.join(inferenceSaveDir, f'{dataset}_****.mat'))['inputFeatures']
        labels_str = os.path.join(rootDir, 'RNNTrainingSteps', '****_HMMLabels', cvPart, f'{dataset}_timeSeriesLabels.mat')
        labels = scipy.io.loadmat(labels_str)
        start_positions = (labels['letterStarts'] / 2).astype(int)
        durations = np.round(labels['letterDurations'] / 2).astype(int)

        for i, sentence in enumerate(sentences[:, 0]):
            eeg_sentence = eeg_data[i]
            for j, char in enumerate(sentence[0]):
                if char in target_characters:
                    start_pos = start_positions[i][j]
                    duration = durations[i][j]
                    signal_segment = eeg_sentence[start_pos:start_pos + duration, :]
                    specdat['segments'].append(signal_segment)
                    specdat['labels'].append(char)

        save_path = os.path.join(group_path, f'{dataset}_special_character.mat')
        scipy.io.savemat(save_path, specdat)
        print(f'Saved processed data to {save_path}')

# 主函数
def main():
    init_setup()
    rootDir = '../handwritingBCIData/'
    dataDirs = ['your_dataset']
    cvPart = 'HeldOutTrials'
    target_characters1 = ['x', 'y']
    target_characters2 = ['r', 'n', 'h']
    process_data(rootDir, dataDirs, cvPart, target_characters1, 1)
    process_data(rootDir, dataDirs, cvPart, target_characters2, 2)

if __name__ == '__main__':
    main()
