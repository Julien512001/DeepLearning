import sys
sys.path.insert(0, '../DeepLearning')

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from PerceptronClass import *
import h5py
import numpy as np
import matplotlib.pyplot as plt




def load_data():
    train_dataset = h5py.File('ChienChat/datasets/trainset.hdf5', "r")
    X_train = np.array(train_dataset["X_train"][:]) # your train set features
    y_train = np.array(train_dataset["Y_train"][:]) # your train set labels

    test_dataset = h5py.File('ChienChat/datasets/testset.hdf5', "r")
    X_test = np.array(test_dataset["X_test"][:]) # your train set features
    y_test = np.array(test_dataset["Y_test"][:]) # your train set labels
    
    return X_train, y_train, X_test, y_test

def CatOrDog(y):
    return np.where(y==0, 'cat', 'Dog')



X_train, y_train, X_test, y_test = load_data()

'''
plt.figure(figsize=(16,8))
for i in range(1,20):
    plt.subplot(4,5,i)
    plt.imshow(X_train[i], cmap='gray')
    plt.title(CatOrDog(y_train)[i]) 
    plt.tight_layout()
plt.show()
'''

X_train_reshaped = X_train.reshape(X_train.shape[0], -1)/np.max(X_train)
X_test_reshaped = X_test.reshape(X_test.shape[0], -1)/np.max(X_test)

P = PerceptronCreator(learning_rate=0.01, n_iter=10000, LossCurve=True)
P.fit(X_train_reshaped, y_train, X_test_reshaped, y_test)


W,b = P.get_params()
print(W,b)
y_pred = P.predict(X_test_reshaped)
score = P.score(y_pred, y_test)
print(score)
