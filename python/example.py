import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import OneClassSVM


# For training dataset
n_samples = 300
outliers_fraction = 0.15
n_outliers = int(outliers_fraction * n_samples)
n_inliers = n_samples - n_outliers

# For test dataset
n_samplesTest = 300
n_outliersTest = int(outliers_fraction * n_samplesTest)
n_inliersTest = n_samplesTest - n_outliersTest

# Colors for plotting the training data
## colors[0] -> noise
## colors[1] -> not noise
colors1 = np.array(["#ff7f00", "#377eb8"])


# Colors for plotting the results of anomaly detection
## colors[0] -> outlier
## colors[1] -> inlier
colors2 = np.array(["red", "blue"])


rng = np.random.RandomState(40)
X_train = np.concatenate((rng.normal(loc=1.0, scale=0.5, size=(n_inliers)), rng.normal(loc=1.0, scale=5.0, size=(n_outliers))))
y = np.hstack((np.ones(n_inliers), -1 * np.ones(n_outliers)))
idx = np.random.permutation(n_samples)
X_train = X_train[idx]
y = y[idx]

# Plotting dataset
plt.figure(figsize=[17, 4])
plt.xlabel("Time (s)")
plt.ylabel("Measurement value")
plt.scatter(range(n_samples), X_train, color=colors1[(y == 1).astype('int')])



clf = OneClassSVM(nu=outliers_fraction, kernel="rbf", gamma=0.1)
clf.fit(X_train.reshape(-1, 1))
y_pred = clf.predict(X_train.reshape(-1, 1))
plt.figure(figsize=[17, 4])
plt.xlabel("Time (s)")
plt.ylabel("Measurement value")
plt.scatter(range(n_samples), X_train, color=colors2[(y_pred == 1).astype('int')])


# Creating test dataset
X_test = np.concatenate((rng.normal(loc=1.0, scale=0.5, size=(n_inliersTest)), rng.normal(loc=1.0, scale=5.0, size=(n_outliersTest))))
yTest = np.hstack((np.ones(n_inliersTest), -1 * np.ones(n_outliersTest)))
idxTest = np.random.permutation(n_samplesTest)
X_test = X_test[idxTest]
yTest = yTest[idxTest]


y_predTest = clf.predict(X_test.reshape(-1, 1))
plt.figure(figsize=[17, 4])
plt.xlabel("Time (s)")
plt.ylabel("Measurement value")
for i in range(n_samplesTest):
    plt.scatter(range(i), X_test[:i], color=colors2[(y_predTest[:i]==1).astype('int')])
    plt.ylim(-12, 12)
    #plt.show()
    #time.sleep(0.5)
    plt.savefig('simulation' + str(i).zfill(5) + '.png')