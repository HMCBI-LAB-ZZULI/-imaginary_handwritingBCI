import numpy as np
import scipy.io
from characterDefinitions import getHandwritingCharacterDefinitions
from dataLabelingStep import labelDataset, constructRNNTargets
import os
import datetime

# point this towards the top level dataset directory
rootDir = '../handwritingBCIData/'

# this line ensures that tensorflow will only use GPU 0 (keeps it from taking over all the GPUs in a multi-gpu setup)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# define which datasets to process
dataDirs = ['choose your dataset']

# defines the list of all 31 characters and what to call them
charDef = getHandwritingCharacterDefinitions()
# saves all labels in this folder
if not os.path.isdir(rootDir + 'RNNTrainingSteps/_HMMLabels'):
    os.mkdir(rootDir + 'RNNTrainingSteps/_HMMLabels')
    for dataDir in dataDirs:
        timeStart = datetime.datetime.now()
        print('Labeling ' + dataDir + ' dataset')

        # load sentences, single letter, time-warped files, and train/test partitions
        sentenceDat = scipy.io.loadmat(rootDir + 'Datasets/' + dataDir + '/sentences.mat')
        singleLetterDat = scipy.io.loadmat(rootDir + 'Datasets/' + dataDir + '/singleLetters.mat')
        twCubes = scipy.io.loadmat(rootDir + 'RNNTrainingSteps/***Warping/' + dataDir + '_warpedCubes.mat')

        cvPart_heldOutBlocks = scipy.io.loadmat(rootDir + 'RNNTrainingSteps/trainTestPartitions_HeldOutBlocks.mat')
        cvPart_heldOutTrials = scipy.io.loadmat(rootDir + 'RNNTrainingSteps/trainTestPartitions_HeldOutTrials.mat')
        cvParts = [cvPart_heldOutBlocks, cvPart_heldOutTrials]
        print("Keys in twCubes:", twCubes.keys())

        # the last two sessions have hashmarks (#) to indicate that T5 should take a brief pause
        # here we remove these from the sentence prompts, otherwise the code below will get confused (because # isn't a character)
        for x in range(sentenceDat['sentencePrompt'].shape[0]):
            sentenceDat['sentencePrompt'][x, 0][0] = sentenceDat['sentencePrompt'][x, 0][0].replace('#', '')

        cvFolderNames = ['HeldOutBlocks', 'HeldOutTrials']

        sentences = sentenceDat['sentencePrompt'][:, 0]
        sentenceLens = sentenceDat['numTimeBinsPerSentence'][:, 0]

        # construct separate labels for each training partition
        for cvPart, cvFolder in zip(cvParts, cvFolderNames):
            print("Labeling '" + cvFolder + "' partition")
            trainPartitionIdx = cvPart[dataDir + '_train']
            testPartitionIdx = cvPart[dataDir + '_test']

            # label the data with an iterative forced alignmnet HMM
            letterStarts, letterDurations, blankWindows = labelDataset(sentenceDat,
                                                                       singleLetterDat,
                                                                       twCubes,
                                                                       trainPartitionIdx,
                                                                       testPartitionIdx,
                                                                       charDef)

            # construct targets for supervised learning
            charStartTarget, charProbTarget, ignoreErrorHere = constructRNNTargets(letterStarts,
                                                                                   letterDurations,
                                                                                   sentenceDat[
                                                                                       'neuralActivityCube'].shape[1],
                                                                                   sentences,
                                                                                   charDef)

            saveDict = {}
            saveDict['letterStarts'] = letterStarts
            saveDict['letterDurations'] = letterDurations
            saveDict['charStartTarget'] = charStartTarget.astype(np.float32)
            saveDict['charProbTarget'] = charProbTarget.astype(np.float32)
            saveDict['ignoreErrorHere'] = ignoreErrorHere.astype(np.float32)
            saveDict['blankWindows'] = blankWindows
            saveDict['timeBinsPerSentence'] = sentenceDat['numTimeBinsPerSentence']

            if not os.path.isdir(rootDir + 'RNNTrainingSteps/_HMMLabels/' + cvFolder):
                os.mkdir(rootDir + 'RNNTrainingSteps/_HMMLabels/' + cvFolder)

            scipy.io.savemat(
                rootDir + 'RNNTrainingSteps/_HMMLabels/' + cvFolder + '/' + dataDir + '_timeSeriesLabels.mat',
                saveDict)

        timeEnd = datetime.datetime.now()
        print('Total time taken: ' + str((timeEnd - timeStart).total_seconds()) + ' seconds')
        print(' ')


