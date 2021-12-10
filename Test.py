import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import plotly.express as px
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split




# Initialisation
# On initialise les paramètres W et b

def init(X):
    W = np.random.randn(X.shape[1], 1)
    b = np.random.randn(1)
    return W,b
# Modèle
# On crée le modèle des neurones artificiels

def model(X,W,b):
    Z = X.dot(W) + b
    A = 1/(1+np.exp(-Z))
    return A

# Cost
# On évalue le coût de notre modèle

def log_loss(A, y):
    m = len(y)
    return 1/m * np.sum(-y * np.log(A) - (1-y) * np.log(1-A))


# Gradients
# Minimiser la fonction coût

def gradients(A, X, y):
    m = len(y)
    dW = 1/m * np.dot(X.T, A-y)
    db = 1/m * np.sum(A-y)
    return (dW, db)


# Update
# On modifie les paramètres pour tendre vers une stabilité

def update(dW, db, W, b, learning_rate):
    W = W - learning_rate*dW
    b = b - learning_rate*db
    return W, b

# Threshold

def predict(X, W, b):
    A = model(X, W, b)
    print(A)
    return A >= 0.5


# On créer maintenant notre Percéptron

def artificial_neuron(X, y, learning_rate = 0.1, n_iter = 100):
    # Initialiser les paramètres W et b
    W, b = init(X)
    y = y.reshape((y.shape[0], 1))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2)

    Loss = []

    for i in range(n_iter):
        A = model(X_train, W, b)
        Loss.append(log_loss(A, y_train))
        dW, db = gradients(A, X_train, y_train)
        W, b = update(dW, db, W, b, learning_rate)
    
    y_pred = predict(X_test, W, b)
    print(accuracy_score(y_test, y_pred))

    
    plt.plot(Loss)
    plt.xlabel("iteration")
    plt.ylabel("error")
    plt.show()

    return W, b


'''
X, y = make_blobs(n_samples=10000, n_features=2, centers=2)

print("dimensions de X : {}".format(X.shape))
print("dimensions de y : {}".format(y.shape))



W, b = artificial_neuron(X,y)
print(W,b)

x0 = np.linspace(-10,10,100)
x1 = (-x0*W[0] - b)/W[1]

plt.scatter(X[:,0], X[:,1], c=y, cmap='summer')
plt.plot(x0, x1, color='orange', lw=3)
plt.show()
'''


