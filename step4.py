import os
import pickle
import logging
import time
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from IPython import display
from scipy.ndimage.filters import gaussian_filter1d
from charSeqRNN import getDefaultRNNArgs


def init_setup():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def create_directories(rootDir, rnnOutputDir):
    paths = [
        os.path.join(rootDir, 'RNNTrainingSteps', '_RNNTraining'),
        os.path.join(rootDir, 'RNNTrainingSteps', '_RNNTraining', rnnOutputDir)
    ]
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)
            logging.info(f'Created directory: {path}')


def configure_rnn_args(rootDir, dataDirs, cvPart):
    args = getDefaultRNNArgs()
    for x, dataDir in enumerate(dataDirs):
        args[f'sentencesFile_{x}'] = os.path.join(rootDir, 'Datasets', dataDir, 'sentences.mat')
        args[f'singleLettersFile_{x}'] = os.path.join(rootDir, 'Datasets', dataDir, 'singleLetters.mat')
        args[f'labelsFile_{x}'] = os.path.join(rootDir, 'RNNTrainingSteps', '_HMMLabels', cvPart,
                                               f'{dataDir}_timeSeriesLabels.mat')
        args[f'syntheticDatasetDir_{x}'] = os.path.join(rootDir, 'RNNTrainingSteps', '_SyntheticSentences', cvPart,
                                                        f'{dataDir}_syntheticSentences')
        args[f'cvPartitionFile_{x}'] = os.path.join(rootDir, 'RNNTrainingSteps', f'trainTestPartitions_{cvPart}.mat')
        args[f'sessionName_{x}'] = dataDir

    args['outputDir'] = os.path.join(rootDir, 'RNNTrainingSteps', '_RNNTraining', cvPart)
    args['dayProbability'] = '[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]'
    args['dayToLayerMap'] = '[0,1,2,3,4,5,6,7,8,9]'
    return args


def save_args(args):
    args_path = os.path.join(args['outputDir'], 'args.p')
    with open(args_path, "wb") as f:
        pickle.dump(args, f)
    logging.info(f'Saved arguments to {args_path}')
    return args_path


def launch_rnn_training(scriptFile, argsFile):
    cmd = f'python {scriptFile} --argsFile={argsFile} &'
    os.system(cmd)
    logging.info(f'Launched RNN training with command: {cmd}')


def visualize_training(args):
    while True:
        try:
            snapshot = scipy.io.loadmat(os.path.join(args['outputDir'], 'outputSnapshot.mat'))
            intOut = scipy.io.loadmat(os.path.join(args['outputDir'], 'intermediateOutput.mat'))
        except FileNotFoundError:
            time.sleep(30)
            continue

        display.clear_output(wait=True)

        plotEnd = np.argwhere(intOut['batchTrainStats'][:, 0] == 0)[1][0] - 1
        plotEndVal = np.argwhere(intOut['batchValStats'][:, 0] == 0)[1][0] - 1

        # Training loss & frame-by-frame accuracy
        plt.figure(figsize=(14, 4))
        plt.subplot(1, 2, 1)
        plt.plot(intOut['batchTrainStats'][0:plotEnd, 0],
                 gaussian_filter1d(intOut['batchTrainStats'][0:plotEnd, 1], 10))
        plt.plot(intOut['batchValStats'][0:plotEndVal, 0],
                 gaussian_filter1d(intOut['batchValStats'][0:plotEndVal, 1], 1))
        plt.plot([0, intOut['batchValStats'][plotEndVal, 0]], [0.50, 0.50], '--k')
        plt.plot([0, intOut['batchValStats'][plotEndVal, 0]], [1.0, 1.0], '--k')
        plt.xlabel('Batch #')
        plt.legend(['Train', 'Test'])
        plt.ylim([0, 3.75])
        plt.ylabel('Loss')

        plt.subplot(1, 2, 2)
        plt.plot(intOut['batchTrainStats'][0:plotEnd, 0],
                 gaussian_filter1d(intOut['batchTrainStats'][0:plotEnd, 3], 10))
        plt.plot(intOut['batchValStats'][0:plotEndVal, 0],
                 gaussian_filter1d(intOut['batchValStats'][0:plotEndVal, 3], 1))
        plt.plot([0, intOut['batchValStats'][plotEndVal, 0]], [0.8, 0.8], '--k')
        plt.plot([0, intOut['batchValStats'][plotEndVal, 0]], [0.9, 0.9], '--k')
        plt.ylim([0, 1.0])
        plt.xlabel('Batch #')
        plt.legend(['Train', 'Test'])
        plt.ylabel('Frame-by-Frame Accuracy')

        plt.suptitle('Training Progress')
        display.display(plt.gcf())
        plt.close()

        # RNN outputs & training targets
        plt.figure(figsize=(12.45, 8.3))
        plt.subplot(2, 2, 1)
        plt.imshow(np.transpose(snapshot['inputs']), aspect='auto', clim=[-1, 1])
        plt.title('Input Features')
        plt.ylabel('Electrode #')
        plt.xlabel('Time Step')

        plt.subplot(2, 2, 2)
        plt.imshow(np.transpose(snapshot['rnnUnits']), aspect='auto', clim=[-1, 1])
        plt.title('RNN Units')
        plt.ylabel('Unit #')
        plt.xlabel('Time Step')

        plt.subplot(2, 2, 3)
        plt.imshow(np.transpose(snapshot['charProbOutput']), aspect='auto')
        plt.title('RNN Probability Outputs (Logits)')
        plt.ylabel('Character #')
        plt.xlabel('Time Step')

        plt.subplot(2, 2, 4)
        plt.imshow(np.transpose(snapshot['charProbTarget']), aspect='auto')
        plt.title('Probability One-Hot Targets')
        plt.ylabel('Character #')
        plt.xlabel('Time Step')

        plt.tight_layout(pad=3)
        plt.suptitle('Inputs & Outputs for Example Snippet')
        display.display(plt.gcf())
        plt.close()

        plt.figure(figsize=(16, 4))
        plt.plot(np.squeeze(snapshot['charStartOutput']))
        plt.plot(np.squeeze(snapshot['charStartTarget']))
        plt.plot(np.squeeze(snapshot['errorWeight']))
        plt.plot([0, snapshot['errorWeight'].shape[1]], [0.3, 0.3], '--k')
        plt.title('Char Start Signal')
        plt.xlabel('Time Step')
        plt.legend(['RNN Output', 'Target', 'Error Weight', 'Threshold'])

        display.display(plt.gcf())
        plt.close()

        time.sleep(30)


def main():
    init_setup()
    rootDir = '../handwritingBCIData/'
    dataDirs = ['choose your dataset']
    cvPart = 'HeldOutTrials'
    rnnOutputDir = cvPart

    create_directories(rootDir, rnnOutputDir)
    args = configure_rnn_args(rootDir, dataDirs, cvPart)
    argsFile = save_args(args)

    scriptFile = os.path.join(os.getcwd(), 'charSeqRNN.py')
    launch_rnn_training(scriptFile, argsFile)

    visualize_training(args)


if __name__ == '__main__':
    main()
