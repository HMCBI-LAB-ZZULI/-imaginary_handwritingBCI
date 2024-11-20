import os
import warnings

import numpy as np
import scipy.io
import scipy.special
import tensorflow as tf
from tensorflow.keras.models import load_model

from characterDefinitions import getHandwritingCharacterDefinitions
from kaldiReadWrite import writeKaldiProbabilityMatrix

# suppress all tensorflow warnings (largely related to compatability with v2)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# point this towards the top level dataset directory
rootDir = '../handwritingBCIData/'

# evaluate the RNN on these datasets
dataDirs = ['choose your dataset']

# use this train/test partition
cvPart = 'HeldOutTrials'

# point this towards the specific RNN we want to evaluate
rnnOutputDir = cvPart

# this prevents tensorflow from taking over more than one gpu on a multi-gpu machine
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# this is where we're going to save the RNN outputs
inferenceSaveDir = os.path.join(rootDir, 'RNNTrainingSteps/_RNNInference', rnnOutputDir)

# Get character definition
charDef = getHandwritingCharacterDefinitions()

def handle_segment(eeg_data, start_idx, end_idx, model, target_info):
    """
    Functions that process segment data
    """
    segment = eeg_data[:, start_idx:end_idx, :]
    if segment.shape[1] < target_info['max_length']:
        padded_segment = np.zeros((segment.shape[0], target_info['max_length'], segment.shape[2]))
        padded_segment[:, :segment.shape[1], :] = segment
        segment = padded_segment
    segment = segment[:, :target_info['max_length'], :]
    prediction = model.predict(segment)
    predicted_label = target_info['class_mapping'][np.argmax(prediction).item()]
    return predicted_label

def evaluateRNNOutput(rnnOutput, rnnInput, numBinsPerSentence, trueText, charDef, charStartThresh=0.3, charStartDelay=15):
    """
    Converts the rnn output (character probabilities & a character start signal) into a discrete sentence and computes
    char/word error rates. Returns error counts and the decoded sentences.
    """
    lgit = rnnOutput[:, :, :-1]
    charStart = rnnOutput[:, :, -1]

    target_characters1 = ['x', 'y']
    target_characters2 = ['r', 'n', 'h']

    model1 = load_model(os.path.join(rootDir, 'RNNTrainingSteps/special_character/group1/siamese_model001.h5'))
    model2 = load_model(os.path.join(rootDir, 'RNNTrainingSteps/special_character/group2/siamese_model05.h5'))

    decStr = decodeCharStr(lgit, charStart, charStartThresh, charStartDelay, numBinsPerSentence, charDef['charListAbbr'])

    allErrCounts = {'charCounts': np.zeros(len(trueText)), 'charErrors': np.zeros(len(trueText)),
                    'wordCounts': np.zeros(len(trueText)), 'wordErrors': np.zeros(len(trueText)),
                    'sentenceCounts': np.zeros(len(trueText)), 'sentenceErrors': np.zeros(len(trueText))}
    allDecSentences = []

    # First pass with model1
    for t in range(len(trueText)):
        thisTrueText = trueText[t, 0][0].replace(' ', '').replace('>', ' ').replace('~', '.').replace('#', '')
        thisDec = decStr[t].replace('>', ' ').replace('~', '.')

        _, mismatches = calculate_edit_distance(list(thisTrueText), list(thisDec))
        eeg_data = np.expand_dims(rnnInput[t, :, :], axis=0)
        letTrans = scipy.special.expit(charStart[t, :])
        endIdx = np.ceil(numBinsPerSentence[t]).astype(int)
        letTrans = letTrans[:endIdx[0]]
        transIdx = np.argwhere(np.logical_and(letTrans[:-1] < charStartThresh, letTrans[1:] > charStartThresh)).squeeze()

        for mismatch in mismatches:
            x, y = mismatch
            if thisTrueText[x - 1] in target_characters1:
                predicted_label = handle_segment(eeg_data, transIdx[y - 1], transIdx[y] if y < len(transIdx) - 1 else endIdx[0],
                                                 model1, target_info={'class_mapping': {0: 'x', 1: 'y'}, 'max_length': 87})
                thisDec = thisDec[:y-1] + predicted_label + thisDec[y:]

        nCharErrors, _ = calculate_edit_distance(list(thisTrueText), list(thisDec))
        nWordErrors, _ = calculate_edit_distance(thisTrueText.strip().split(), thisDec.strip().split())
        nSentenceErrors = 0 if thisTrueText == thisDec else 1

        allErrCounts['charCounts'][t] = len(thisTrueText)
        allErrCounts['charErrors'][t] = nCharErrors
        allErrCounts['wordCounts'][t] = len(thisTrueText.strip().split())
        allErrCounts['wordErrors'][t] = nWordErrors
        allErrCounts['sentenceCounts'][t] = 1
        allErrCounts['sentenceErrors'][t] = nSentenceErrors
        allDecSentences.append(thisDec)

    # Second pass with model2
    for t in range(len(trueText)):
        thisTrueText = trueText[t, 0][0].replace(' ', '').replace('>', ' ').replace('~', '.').replace('#', '')
        thisDec = allDecSentences[t].replace('>', ' ').replace('~', '.')

        _, mismatches = calculate_edit_distance(list(thisTrueText), list(thisDec))
        eeg_data = np.expand_dims(rnnInput[t, :, :], axis=0)
        letTrans = scipy.special.expit(charStart[t, :])
        endIdx = np.ceil(numBinsPerSentence[t]).astype(int)
        letTrans = letTrans[:endIdx[0]]
        transIdx = np.argwhere(np.logical_and(letTrans[:-1] < charStartThresh, letTrans[1:] > charStartThresh)).squeeze()

        for mismatch in mismatches:
            x, y = mismatch
            if thisTrueText[x - 1] in target_characters2:
                predicted_label = handle_segment(eeg_data, transIdx[y - 1], transIdx[y] if y < len(transIdx) - 1 else endIdx[0],
                                                 model2,  target_info={'class_mapping': {0: 'r', 1: 'n', 2: 'h'}, 'max_length': 75})
                thisDec = thisDec[:y-1] + predicted_label + thisDec[y:]

        nCharErrors, _ = calculate_edit_distance(list(thisTrueText), list(thisDec))
        nWordErrors, _ = calculate_edit_distance(thisTrueText.strip().split(), thisDec.strip().split())
        nSentenceErrors = 0 if thisTrueText == thisDec else 1

        allErrCounts['charCounts'][t] = len(thisTrueText)
        allErrCounts['charErrors'][t] = nCharErrors
        allErrCounts['wordCounts'][t] = len(thisTrueText.strip().split())
        allErrCounts['wordErrors'][t] = nWordErrors
        allErrCounts['sentenceCounts'][t] = 1
        allErrCounts['sentenceErrors'][t] = nSentenceErrors
        allDecSentences[t] = thisDec

    return allErrCounts, allDecSentences

def decodeCharStr(logitMatrix, transSignal, transThresh, transDelay, numBinsPerTrial, charList):
    """
    Converts the rnn output (character probabilities & a character start signal) into a discrete sentence.
    """
    decWords = []
    for v in range(logitMatrix.shape[0]):
        logits = np.squeeze(logitMatrix[v, :, :])
        bestClass = np.argmax(logits, axis=1)
        letTrans = scipy.special.expit(transSignal[v, :])

        endIdx = np.ceil(numBinsPerTrial[v]).astype(int)
        letTrans = letTrans[:endIdx[0]]

        transIdx = np.argwhere(np.logical_and(letTrans[:-1] < transThresh, letTrans[1:] > transThresh)).squeeze()

        wordStr = ''
        for x in range(len(transIdx)):
            wordStr += charList[bestClass[transIdx[x] + transDelay]]

        decWords.append(wordStr)

    return decWords

def calculate_edit_distance(reference, prediction):
    """
    Computes the edit distance between two sequences and returns the distance and mismatch positions.
    """
    m, n = len(reference), len(prediction)
    d = np.zeros((m + 1, n + 1), dtype=np.uint8)

    for i in range(m + 1):
        d[i][0] = i
    for j in range(n + 1):
        d[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if reference[i - 1] == prediction[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    mismatch_positions = []
    i, j = m, n
    while i > 0 and j > 0:
        if reference[i - 1] == prediction[j - 1]:
            i -= 1
            j -= 1
        else:
            if d[i][j] == d[i - 1][j - 1] + 1 and d[i][j - 1] == d[i - 1][j]:
                mismatch_positions.append((i, j))
                i -= 1
                j -= 1
            elif d[i][j - 1] > d[i - 1][j]:
                i -= 1
            else:
                mismatch_positions.append((i, j))
                j -= 1

    mismatch_positions.reverse()
    return d[m][n], mismatch_positions

def rnnOutputToKaldiMatrices(rnnOutput, numBinsPerSentence, charDef, kaldiDir):
    """
    Converts the rnn output into probability matrices that Kaldi can read, one for each sentence.
    As part of the conversion, this function creates a CTC blank signal from the character start signal so
    that the language model is happy (it was designed for a CTC loss).
    """
    lgit = rnnOutput[:, :, :-1]
    charProb = np.exp(lgit) / np.sum(np.exp(lgit), axis=2, keepdims=True)
    charStart = rnnOutput[:, :, -1]

    fakeCTC = np.ones(charStart.shape)
    fakeCTC[:, 20:] = 1 - scipy.special.expit(4 + 4 * charStart[:, :-20])

    nChar = rnnOutput.shape[2] - 1
    probCombined = np.concatenate([charProb, fakeCTC[:, :, np.newaxis]], axis=2)
    probCombined[:, :, :nChar] *= 1 - fakeCTC[:, :, np.newaxis]

    allMatrices = []
    for t in range(rnnOutput.shape[0]):
        startIdx = 0
        endIdx = int(numBinsPerSentence[t, 0])
        charProb = np.transpose(probCombined[t, startIdx:endIdx:5, charDef['idxToKaldi']])

        charProb[charProb == 0] = 1e-13
        charProb = np.log(charProb)

        writeKaldiProbabilityMatrix(charProb, t, os.path.join(kaldiDir, f'kaldiMat_{t}.txt'))
        allMatrices.append(charProb)

    return allMatrices

# Main evaluation script
warnings.simplefilter(action='ignore', category=FutureWarning)

allErrCounts = []

for dataDir in dataDirs:
    print('-- ' + dataDir + ' --')

    # Load RNN outputs and inputs
    outputs = scipy.io.loadmat(os.path.join(inferenceSaveDir, f'{dataDir}_inferenceOutputs.mat'))
    inputs = scipy.io.loadmat(os.path.join(rootDir, 'RNNTrainingSteps/_RNNInference', rnnOutputDir, f'{dataDir}_inferenceinputFeatures.mat'))
    sentenceDat = scipy.io.loadmat(os.path.join(rootDir, 'Datasets', dataDir, 'sentences.mat'))

    # Convert the outputs into character sequences & get word/character error counts
    errCounts, decSentences = evaluateRNNOutput(outputs['outputs'],
                                                inputs['inputFeatures'],
                                                sentenceDat['numTimeBinsPerSentence'] / 2 + 50,
                                                sentenceDat['sentencePrompt'],
                                                charDef,
                                                charStartThresh=0.30,
                                                charStartDelay=15)

    # Save decoded sentences and error counts
    saveDict = {'decSentences': decSentences, 'trueSentences': sentenceDat['sentencePrompt']}
    saveDict.update(errCounts)
    scipy.io.savemat(os.path.join(inferenceSaveDir, f'{dataDir}_errCounts.mat'), saveDict)

    # Print results for the validation sentences
    cvPartFile = scipy.io.loadmat(os.path.join(rootDir, f'RNNTrainingSteps/trainTestPartitions_{cvPart}.mat'))
    valIdx = cvPartFile[f'{dataDir}_test']

    if len(valIdx) == 0:
        print('No validation sentences for this session.')
        print('  ')
        continue

    valAcc = 100 * (1 - np.sum(errCounts['charErrors'][valIdx]) / np.sum(errCounts['charCounts'][valIdx]))
    w_valAcc = 100 * (1 - np.sum(errCounts['wordErrors'][valIdx]) / np.sum(errCounts['wordCounts'][valIdx]))
    s_valAcc = 100 * (1 - np.sum(errCounts['sentenceErrors'][valIdx]) / np.sum(errCounts['sentenceCounts'][valIdx]))

    print(f'Character error rate for this session: {100 - valAcc:.2f}%')
    print(f'Word error rate for this session: {100 - w_valAcc:.2f}%')
    print(f'Sentence error rate for this session: {100 - s_valAcc:.2f}%')
    print(' ')

    # Append error counts for overall summary
    allErrCounts.append(np.stack([
        errCounts['charCounts'][valIdx],
        errCounts['charErrors'][valIdx],
        errCounts['wordCounts'][valIdx],
        errCounts['wordErrors'][valIdx],
        errCounts['sentenceCounts'][valIdx],
        errCounts['sentenceErrors'][valIdx]
    ], axis=0).T)

# Summarize character, word, and sentence error rates across all sessions
concatErrCounts = np.squeeze(np.concatenate(allErrCounts, axis=0))
cer = 100 * (np.sum(concatErrCounts[:, 1]) / np.sum(concatErrCounts[:, 0]))
wer = 100 * (np.sum(concatErrCounts[:, 3]) / np.sum(concatErrCounts[:, 2]))
ser = 100 * (np.sum(concatErrCounts[:, 5]) / np.sum(concatErrCounts[:, 4]))

print(f'Character error rate: {cer:.2f}%')
print(f'Word error rate: {wer:.2f}%')
print(f'Sentence error rate: {ser:.2f}%')
