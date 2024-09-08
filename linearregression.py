import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets



def hypothesis(w, X, b):
    return np.dot(X,w) + b

def cost_function(w, X, b, y):
    n_samples, n_features= X.shape
    cost= 1/(2*n_samples) * (np.sum(hypothesis(w,X,b)-y) ** 2)
    return cost

def gradient_desc(w, X, b, y, rate, threshold):
    n_samples, n_features= X.shape
    i=0
    converged= False
    while not converged:
        y_predicted= hypothesis(w, X, b)
        w_old=w
        b_old=b
        dw= 1/n_samples * np.dot(X.T, (y_predicted - y))
        db= 1/n_samples * np.sum(y_predicted - y)
        w= w_old - (rate * dw)
        b= b_old - (rate * db)
        if np.all(np.abs(w - w_old) <= threshold) and np.abs(b - b_old) <= threshold:
             converged=True



    return w, b

def linear_regression(X, y, learning_rate=0.01, threshold=0.001):
    n_samples, n_features= X.shape
    w = np.zeros(n_features)
    b = 0
    w, b= gradient_desc(w, X, b, y, learning_rate, threshold)

    return w, b

def predict(w, X, b):
    y_predicted= hypothesis(w, X, b)
    return y_predicted


iris = datasets.load_iris()

X = iris.data[:, :1]  #sepal length as input
y = iris.data[:, 1]   #sepal width as target 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

w, b= linear_regression(X_train, y_train)
print(f"weight: {w}")
print(f"bias: {b}")
y_prediction_line= predict(w, X, b)
cmap= plt.get_cmap('viridis')
fig= plt.figure(figsize=(6, 6))
m1= plt.scatter(X_train, y_train, color=cmap(0.2), s=10)
m2= plt.scatter(X_test, y_test, color=cmap(1.0), s=10)
plt.plot(X, y_prediction_line, color='black', linewidth=2, label= 'prediction')
plt.show()


    
    