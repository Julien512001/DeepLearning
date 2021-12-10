from scipy.sparse.linalg.isolve.iterative import qmr
from PerceptronClass import *
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split

X, y = make_blobs(n_samples = 1000,n_features= 2, centers=2, random_state=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)
y_train, y_test = y_train.reshape((y_train.shape[0], 1)), y_test.reshape((y_test.shape[0], 1))



P = PerceptronCreator(learning_rate=0.01, n_iter=1000, LossCurve=True, fit_curve=True)
P.fit(X_train, y_train)
W,b = P.get_params()


y_pred = P.predict(X_test)
score = P.score(y_pred, y_test)
print(score)


