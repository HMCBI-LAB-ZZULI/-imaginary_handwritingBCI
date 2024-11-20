import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import scipy.io

rootDir = '../handwritingBCIData/'
mydata = scipy.io.loadmat(rootDir + 'your source')
# Loading data
data = mydata['data']  # (837, -1) 

# Combine the data and labels into one large matrix
labels_all = np.concatenate([[i] * 27 for i in range(31)])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels_all, test_size=0.2, random_state=41)

# Create KNN model
knn = KNeighborsClassifier()

# Perform K-fold cross-validation with k=5
scores = cross_val_score(knn, X_train, y_train, cv=10)

# Print the mean and standard deviation of the accuracy scores
print('Accuracy scores:', scores)
print('Mean accuracy:', np.mean(scores))
print('Standard deviation:', np.std(scores))
