import numpy as np
import pandas as pd
import pylab
from sklearn.metrics import mean_squared_error
from sklearn import linear_model

if __name__ == '__main__':
    data = pd.read_csv('dip-har-eff.csv')
    
    rows = data.shape [0]
    cols = data.shape [1]
    data = data.values
    data = data[np.arange(0,rows),:]
    X = data[:, 1]
    Y = data[:, 2]
    X_max = np.max(X)
    Y_max = np.max(Y)
    X = np.true_divide (X, X_max)
    Y = np.true_divide (Y, Y_max)
    pylab.xlim(0,max(X))
    pylab.ylim(0,max(Y))
 
    mu = 0.1 # learning rate
    epochs = 1000
    b1 = 0
    b0 = 0
    batch_size = 15

    for i in np.arange(epochs):
        Y_pred = b1*X + b0  # The current predicted value of Y
        D_m = (-2/batch_size) * sum(X * (Y - Y_pred))  # Derivative wrt m
        D_c = (-2/batch_size) * sum(Y - Y_pred)  # Derivative wrt c
        E = mean_squared_error(Y, Y_pred)
        b1 = b1 - mu * D_m  # Update m
        b0 = b0 - mu * D_c  # Update c

    for i in np.arange(epochs):
        Y_pred = b1*X + b0
        E = mean_squared_error(Y, Y_pred)

    print (" b0: " + str(b0) + " b1: " + str(b1) + " Error: " + str(E))

    pylab.plot(X, b1*X + b0)

    # plot
    for i in range(X.shape[0]):
        y_pred = b1*X + b0 

    pylab.plot(X,Y,'o')
    pylab.show()


    # predicted value
    data = pd.read_csv("set-har-eff.csv")

    X_set = data[['year', 'effort']]
    y_set = data['harvest']

    regr = linear_model.LinearRegression()
    regr.fit(X_set, y_set)

    predicted_harvest = regr.predict([[2023, 28375]])

    #print(predicted_harvest)

    