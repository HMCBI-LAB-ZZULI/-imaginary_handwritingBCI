import os
import tensorflow as tf
import numpy as np
import scipy.io
import scipy.ndimage.filters
import matplotlib.pyplot as plt
from sklearn.decomposition import IncrementalPCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from twpca import TWPCA
from twpca.regularizers import curvature
from characterDefinitions import getHandwritingCharacterDefinitions


np.random.seed(42)

# Setting the data path
rootDir = '../handwritingBCIData/'
dataDirs = ['choose your dataset']
charDef = getHandwritingCharacterDefinitions()

#Create a save directory
outputDir = os.path.join(rootDir, 'RNNTrainingSteps', '(test)_TimeWarping')
if not os.path.isdir(outputDir):
    os.mkdir(outputDir)

for dataDir in dataDirs:
    print('Warping dataset:', dataDir)
    dat = scipy.io.loadmat(os.path.join(rootDir, 'Datasets', dataDir, 'singleLetters.mat'))

    #Normalize data
    trlIdx_all = []
    for char in charDef['charList']:
        neuralCube = dat['neuralActivityCube_' + char].astype(np.float64)
        trlIdx = [t for t in range(dat['characterCues'].shape[0]) if dat['characterCues'][t, 0] == char]
        trlIdx_all.extend(trlIdx)
        blockIdx = dat['blockNumsTimeSeries'][dat['goPeriodOnsetTimeBin'][trlIdx]].squeeze()
        for b in range(dat['blockList'].shape[0]):
            trialsFromThisBlock = np.squeeze(blockIdx == dat['blockList'][b])
            neuralCube[trialsFromThisBlock, :, :] -= dat['meansPerBlock'][np.newaxis, b, :]
        neuralCube /= dat['stdAcrossAllData'][np.newaxis, :, :]
        dat['neuralActivityCube_' + char] = neuralCube

    alignedDat = {}
    data = []
    ipca = IncrementalPCA(n_components=5)

    for i, char in zip(range(31), charDef['charList']):
        print('Warping character:', char)
        tf.reset_default_graph()

        smoothed_spikes = scipy.ndimage.filters.gaussian_filter1d(dat['neuralActivityCube_' + char], 4.0, axis=1)
        model = TWPCA(smoothed_spikes, n_components=5,
                      warp_regularizer=curvature(scale=0.001, power=1),
                      time_regularizer=curvature(scale=1.0, power=2, axis=0)).fit(progressbar=False)
        estimated_aligned_data = model.transform(dat['neuralActivityCube_' + char])
        smoothed_aligned_data = scipy.ndimage.filters.gaussian_filter1d(estimated_aligned_data, 4.0, axis=1)

        alignedDat[char] = estimated_aligned_data
        Nan_data = np.nan_to_num(smoothed_aligned_data)
        if len(data) <= i:
            data.append(Nan_data)
        else:
            data[i] = Nan_data
        data[i] = np.reshape(data[i], (27, -1))
        ipca.partial_fit(data[i])

    data_pca = np.vstack([ipca.transform(data[i]) for i in range(31)])
    tsne = TSNE(n_components=2, perplexity=20, random_state=0)
    data_tsne = tsne.fit_transform(data_pca)

    outputPath = os.path.join(outputDir, f'{dataDir}_warpedCubes.mat')
    print('Saving', outputPath)
    scipy.io.savemat(outputPath, {'data': data_tsne})

# Loading and preparing data for KNN cross validation
dataPath = os.path.join(outputDir, 't5.2019.05.08_warpedCubes.mat')
mydata = scipy.io.loadmat(dataPath)
data = mydata['data']

labels_all = np.concatenate([[i] * 27 for i in range(31)])
X_train, X_test, y_train, y_test = train_test_split(data, labels_all, test_size=0.2, random_state=41)

knn = KNeighborsClassifier()
scores = cross_val_score(knn, X_train, y_train, cv=10)
print('Accuracy scores:', scores)
print('Mean accuracy:', np.mean(scores))
print('Standard deviation:', np.std(scores))

#Visualizing t-SNE Results
colors = [
    'red', 'blue', 'green', 'purple', 'orange', 'yellow', 'black', 'gray', 'pink', 'brown', 'magenta'
]

fig, ax = plt.subplots()
for i, char in zip(range(31), charDef['charListAbbr']):
    color = colors[i % len(colors)]
    idx = np.where(labels_all == i)
    ax.scatter(data[idx, 0], data[idx, 1], color=color, s=5, label=char)
    ax.annotate(char, (data[idx[0][-1], 0], data[idx[0][-1], 1]), color=color)

plt.title('t-SNE Visualization of PCA-reduced Data')
plt.legend(loc='best', markerscale=2, fontsize='small')
plt.show()
