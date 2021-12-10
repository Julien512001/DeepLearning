import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px

class PerceptronCreator:

    def __init__(self, learning_rate=0.1, n_iter=100, LossCurve=False, class_names = None, fit_curve = False) -> None:
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.LossCurve = LossCurve
        self.class_names = class_names
        self.fit_curve = fit_curve
    
    def initialisation(self,X):
        W = np.random.randn(X.shape[1], 1)
        b = np.random.randn(1)
        return W,b
    
    def model(self,X,W,b):
        Z = X.dot(W) + b
        A = 1 / (1 + np.exp(-Z))
        return A
    
    def log_loss(self,A, y):
        return 1 / len(y) * np.sum(-y * np.log(A) - (1 - y) * np.log(1 - A))

    def gradients(self,A, X, y):
        m = len(y)
        dW = 1/m * np.dot(X.T, A-y)
        db = 1/m * np.sum(A-y)
        return (dW, db)
    
    def update(self,dW, db, W, b, learning_rate):
        W = W - learning_rate*dW
        b = b - learning_rate*db
        return W,b

    
    def predict(self, X):
        A = self.model(X, W, b)
        if (self.class_names == None):
            return 1*(A >= 0.5).ravel()
        return np.where(A >= 0.5, self.class_names[0], self.class_names[1]).ravel()
    
    def score(self, y_pred, y_true):
        return accuracy_score(y_true, y_pred)


    def fit(self, X, y):
        if(len(X) <= 1): 
            raise Exception("Sorry, scalars are not compatible with this model")
        global W
        global b
        W,b = self.initialisation(X)
        

        
        Loss = []

        for i in range(self.n_iter):
            A = self.model(X, W, b)
            Loss.append(self.log_loss(A, y))
            dW, db = self.gradients(A, X, y)
            W, b = self.update(dW, db, W, b, self.learning_rate)
            
        if (self.LossCurve):
            plt.figure()
            plt.plot(Loss)
            plt.xlabel("iteration")
            plt.ylabel("error")
            
        if (self.fit_curve and X.shape[1] == 2):
            plt.figure()
            plt.scatter(X[:,0], X[:,1], c=y.ravel(),cmap='magma')
            plt.xlabel('First feature')
            plt.ylabel('second feature')
            min = np.minimum(np.min(X[:,0]), np.min(X[:,1]))
            max = np.maximum(np.max(X[:,0]),np.max(X[:,1]))
            x0 = np.linspace(min,max,100)
            x1 = (-W[0]*x0-b)/W[1]
            plt.plot(x0, x1, c='red', lw=3)
        
        plt.show()

        
    def get_params(self):
        return W,b
