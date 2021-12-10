import h5py
import numpy as np
import matplotlib.pyplot as plt


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

def view(X,y):
    y_name = cat_Or_dog(y)
    plt.figure(figsize=(16,8))
    for i in range(1,10):
        plt.subplot(4,5,i)
        plt.imshow(X[i], cmap='gray')
        plt.title(y_name) 
        plt.tight_layout()
    plt.show()

