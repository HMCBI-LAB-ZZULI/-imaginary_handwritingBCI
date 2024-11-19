import numpy as np
import scipy.special
import tensorflow as tf

from kaldiReadWrite import writeKaldiProbabilityMatrix

# point this towards the top level dataset directory
rootDir = '../handwritingBCIData/'

def handle_segment(eeg_data, start_idx, end_idx, model, target_info):
    """
    处理 segment 数据的函数
    """
    # 处理长度不足的情况，补零填充或截断
    segment = eeg_data[:, start_idx:end_idx, :]
    if segment.shape[1] < target_info['max_length']:
        padded_segment = np.zeros((segment.shape[0], target_info['max_length'], segment.shape[2]))
        padded_segment[:, :segment.shape[1], :] = segment
        segment = padded_segment
    # 如果长度超过 max_length，截断
    segment = segment[:, :target_info['max_length'], :]
    prediction = model.predict(segment)
    predicted_label = target_info['class_mapping'][np.argmax(prediction).item()]
    return predicted_label

def evaluateRNNOutput(rnnOutput, rnninput , numBinsPerSentence, trueText, charDef, charStartThresh=0.3,
                      charStartDelay=15):
    """
    Converts the rnn output (character probabilities & a character start signal) into a discrete sentence and computes
    char/word error rates. Returns error counts and the decoded sentences.
    """
    lgit = rnnOutput[:, :, 0:-1]
    charStart = rnnOutput[:, :, -1]

    target_characters1 = ['x', 'y']  # 目标字符列表
    target_characters2 = ['r', 'n', 'h']

    model1 = tf.keras.models.load_model(rootDir + 'RNNTrainingSteps/special_character/group1/siamese_model001.h5')
    model2 = tf.keras.models.load_model(rootDir + 'RNNTrainingSteps/special_character/group2/siamese_model05.h5')

    # convert output to character strings
    decStr = decodeCharStr(lgit, charStart, charStartThresh, charStartDelay,
                           numBinsPerSentence, charDef['charListAbbr'])

    allErrCounts = {}
    allErrCounts['charCounts'] = np.zeros([len(trueText)])
    allErrCounts['charErrors'] = np.zeros([len(trueText)])
    allErrCounts['wordCounts'] = np.zeros([len(trueText)])
    allErrCounts['wordErrors'] = np.zeros([len(trueText)])
    allErrCounts['sentenceCounts'] = np.zeros([len(trueText)])
    allErrCounts['sentenceErrors'] = np.zeros([len(trueText)])
    allDecSentences = []

    # First loop with model1 and target_characters1 (x and y)
    for t in range(len(trueText)):
        thisTrueText = trueText[t, 0][0].replace(' ', '').replace('>', ' ').replace('~', '.').replace('#', '')
        thisDec = decStr[t].replace('>', ' ').replace('~', '.')

        _, mismatches = calculate_edit_distance(list(thisTrueText), list(thisDec))
        eeg_data = np.expand_dims(rnninput[t, :, :], axis=0)
        letTrans = scipy.special.expit(charStart[t, :])
        endIdx = np.ceil(numBinsPerSentence[t]).astype(int)
        letTrans = letTrans[0:endIdx[0]]
        transIdx = np.argwhere(np.logical_and(letTrans[0:-1] < charStartThresh, letTrans[1:] > charStartThresh))
        transIdx = transIdx[:, 0]

        for mismatch in mismatches:
            x = mismatch[0]  # 获取x坐标
            y = mismatch[1]  # 获取y坐标
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

    # Second loop with model2 and target_characters2 (r, n, and h)
    for t in range(len(trueText)):
        thisTrueText = trueText[t, 0][0].replace(' ', '').replace('>', ' ').replace('~', '.').replace('#', '')
        thisDec = allDecSentences[t].replace('>', ' ').replace('~', '.')

        _, mismatches = calculate_edit_distance(list(thisTrueText), list(thisDec))
        eeg_data = np.expand_dims(rnninput[t, :, :], axis=0)
        letTrans = scipy.special.expit(charStart[t, :])
        endIdx = np.ceil(numBinsPerSentence[t]).astype(int)
        letTrans = letTrans[0:endIdx[0]]
        transIdx = np.argwhere(np.logical_and(letTrans[0:-1] < charStartThresh, letTrans[1:] > charStartThresh))
        transIdx = transIdx[:, 0]

        for mismatch in mismatches:
            x = mismatch[0]  # 获取x坐标
            y = mismatch[1]  # 获取y坐标
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
        letTrans = letTrans[0:endIdx[0]]

        transIdx = np.argwhere(np.logical_and(letTrans[0:-1] < transThresh, letTrans[1:] > transThresh))
        transIdx = transIdx[:, 0]

        wordStr = ''
        for x in range(len(transIdx)):
            wordStr += charList[bestClass[transIdx[x] + transDelay]]

        decWords.append(wordStr)

    return decWords


def calculate_edit_distance(reference, prediction):
    m = len(reference)
    n = len(prediction)
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
    i = m
    j = n
    while i > 0 and j > 0:
        if reference[i - 1] == prediction[j - 1]:
            i -= 1
            j -= 1
        else:
            if d[i][j] == d[i - 1][j - 1] + 1 and d[i][j - 1] == d[i - 1][j]:  # 替换错误
                mismatch_positions.append((i, j))
                i -= 1
                j -= 1
            elif d[i][j - 1] > d[i - 1][j]:  # 删除错误或相同
                i -= 1
            elif d[i][j - 1] < d[i - 1][j]:  # 插入错误或相同
                mismatch_positions.append((i, j))
                j -= 1
            else:
                j -= 1

    mismatch_positions.reverse()  # Reverse the list to get the positions in order

    return d[m][n], mismatch_positions


def rnnOutputToKaldiMatrices(rnnOutput, numBinsPerSentence, charDef, kaldiDir):
    """
    Converts the rnn output into probability matrices that Kaldi can read, one for each sentence.
    As part of the conversion, this function creates a CTC blank signal from the character start signal so
    that the language model is happy (it was designed for a CTC loss). 
    """
    lgit = rnnOutput[:,:,0:-1]
    charProb = np.exp(lgit)/np.sum(np.exp(lgit),axis=2,keepdims=True)
    charStart = rnnOutput[:,:,-1]

    fakeCTC = np.ones(charStart.shape)
    fakeCTC[:,20:] = 1-scipy.special.expit(4 + 4*charStart[:,0:-20])
    
    nChar = rnnOutput.shape[2]-1
    probCombined = np.concatenate([charProb, fakeCTC[:,:,np.newaxis]],axis=2)
    probCombined[:,:,0:nChar] *= 1-fakeCTC[:,:,np.newaxis]
    
    allMatrices = []
    for t in range(rnnOutput.shape[0]):
        startIdx = 0
        endIdx = int(numBinsPerSentence[t,0])
        charProb = np.transpose(probCombined[t,startIdx:endIdx:5,charDef['idxToKaldi']])

        charProb[charProb==0] = 1e-13
        charProb = np.log(charProb)

        writeKaldiProbabilityMatrix(charProb, t, kaldiDir + 'kaldiMat_'+str(t)+'.txt')
        allMatrices.append(charProb)
        
    return allMatrices