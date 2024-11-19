import tensorflow as tf
import scipy.io
import os
import multiprocessing
import datetime
import logging
from characterDefinitions import getHandwritingCharacterDefinitions
from makeSyntheticSentences import generateCharacterSequences, extractCharacterSnippets, addSingleLetterSnippets
from dataPreprocessing import normalizeSentenceDataCube


# 初始化设置
def init_setup():
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# 处理每个数据集
def process_dataset(rootDir, dataDir, cvParts, charDef):
    logging.info(f'Processing {dataDir}')

    for cvPart in cvParts:
        logging.info(f'-- {cvPart}')

        try:
            sentenceDat = scipy.io.loadmat(os.path.join(rootDir, 'Datasets', dataDir, 'sentences.mat'))
            singleLetterDat = scipy.io.loadmat(os.path.join(rootDir, 'Datasets', dataDir, 'singleLetters.mat'))
            twCubes = scipy.io.loadmat(
                os.path.join(rootDir, 'RNNTrainingSteps', '_TimeWarping', f'{dataDir}_warpedCubes.mat'))

            cvPartFile = scipy.io.loadmat(
                os.path.join(rootDir, 'RNNTrainingSteps', f'trainTestPartitions_{cvPart}.mat'))
            trainPartitionIdx = cvPartFile[f'{dataDir}_train']

            # 移除句子中的hashmarks (#)
            for x in range(sentenceDat['sentencePrompt'].shape[0]):
                sentenceDat['sentencePrompt'][x, 0][0] = sentenceDat['sentencePrompt'][x, 0][0].replace('#', '')

            neuralCube = normalizeSentenceDataCube(sentenceDat, singleLetterDat)
            labels = scipy.io.loadmat(
                os.path.join(rootDir, 'RNNTrainingSteps', '_HMMLabels', cvPart, f'{dataDir}_timeSeriesLabels.mat'))

            snippetDict = extractCharacterSnippets(
                labels['letterStarts'], labels['blankWindows'], neuralCube,
                sentenceDat['sentencePrompt'][:, 0], sentenceDat['numTimeBinsPerSentence'][:, 0],
                trainPartitionIdx, charDef
            )
            snippetDict = addSingleLetterSnippets(snippetDict, singleLetterDat, twCubes, charDef)

            save_dir = os.path.join(rootDir, 'RNNTrainingSteps', '_SyntheticSentences', cvPart)
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            save_path = os.path.join(save_dir, f'{dataDir}_snippets.mat')
            scipy.io.savemat(save_path, snippetDict)

        except Exception as e:
            logging.error(f'Error processing {dataDir} in {cvPart}: {e}')
            continue


# 生成合成数据
def generate_synthetic_data(rootDir, dataDir, cvParts, nParallelProcesses):
    logging.info(f'Generating synthetic data for {dataDir}')

    for cvPart in cvParts:
        logging.info(f'-- {cvPart}')

        outputDir = os.path.join(rootDir, 'RNNTrainingSteps', '_SyntheticSentences', cvPart,
                                 f'{dataDir}_syntheticSentences')
        bashDir = os.path.join(rootDir, 'bashScratch')
        repoDir = os.getcwd()

        os.makedirs(outputDir, exist_ok=True)
        os.makedirs(bashDir, exist_ok=True)

        args = {
            'nSentences': 256,
            'nSteps': 2400,
            'binSize': 2,
            'wordListFile': os.path.join(repoDir, 'wordList', 'google-10000-english-usa.txt'),
            'rareWordFile': os.path.join(repoDir, 'wordList', 'rareWordIdx.mat'),
            'snippetFile': os.path.join(rootDir, 'RNNTrainingSteps', '_SyntheticSentences', cvPart,
                                        f'{dataDir}_snippets.mat'),
            'accountForPenState': 1,
            'charDef': getHandwritingCharacterDefinitions(),
            'seed': datetime.datetime.now().microsecond
        }

        argList = []
        for x in range(20):
            newArgs = args.copy()
            newArgs['saveFile'] = os.path.join(outputDir, f'bat_{x}.tfrecord')
            newArgs['seed'] += x
            argList.append(newArgs)

        with multiprocessing.Pool(nParallelProcesses) as pool:
            results = pool.map(generateCharacterSequences, argList)


def main():
    init_setup()
    rootDir = '../handwritingBCIData/'
    dataDirs = ['choose your dataset']
    cvParts = ['HeldOutBlocks', 'HeldOutTrials']
    charDef = getHandwritingCharacterDefinitions()

    for dataDir in dataDirs:
        process_dataset(rootDir, dataDir, cvParts, charDef)
        generate_synthetic_data(rootDir, dataDir, cvParts, nParallelProcesses=10)


if __name__ == '__main__':
    main()
