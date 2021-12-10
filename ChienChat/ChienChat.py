import numpy as np
import h5py
import matplotlib.pyplot as plt
from Test import *

def load_data():
    train_dataset = h5py.File('datasets/trainset.hdf5', "r")
    X_train = np.array(train_dataset["X_train"][:]) # your train set features
    y_train = np.array(train_dataset["Y_train"][:]) # your train set labels

    test_dataset = h5py.File('datasets/testset.hdf5', "r")
    X_test = np.array(test_dataset["X_test"][:]) # your train set features
    y_test = np.array(test_dataset["Y_test"][:]) # your train set labels
    
    return X_train, y_train, X_test, y_test

def cat_Or_dog(y):
    return np.where(y==0, 'cat', 'Dog')

def normalize(X):
    X_return = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            for k in range(X.shape[2]):
                X_return[i,j,k] = X[i,j,k]/255
    return X_return

def flat(X):
    X_return = np.zeros((X.shape[0], X.shape[1]*X.shape[2]))
    for i in range(X.shape[0]):
        X_return[i] = X[i].flatten()
    return X_return

X_train, y_train, X_test, y_test = load_data()
y_label = cat_Or_dog(y_train)


plt.figure(figsize=(16,8))
for i in range(1,10):
    plt.subplot(4,5,i)
    plt.imshow(X_train[i], cmap='gray')
    plt.title(y_label[i]) 
    plt.tight_layout()
plt.show()



X_train = normalize(X_train)
X_test = normalize(X_test)

X_train = flat(X_train)
X_test = flat(X_test)

print(X_train[0])

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


W, b = artificial_neuron(X_train, y_train)
y_pred = predict(X_test,W,b)
score = accuracy_score(y_test, y_pred)
print(score)

