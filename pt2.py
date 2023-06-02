import pandas
from sklearn import linear_model

data = pandas.read_csv("albacore_metal.dat")

X = data[['mercury', 'lead']]
y = data['cadmium']

regr = linear_model.LinearRegression()
regr.fit(X, y)

print(regr.coef_)