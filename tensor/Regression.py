#Importing Modules/Packages
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style
#Loading in Our Data
data = pd.read_csv("C:/Users/celes/Documents/MachineLearning/tensor/student-mat.csv", sep = ";")
print(data.head(5))
#Trimming Our Data
data = data[["G1", "G2", "G3",  "studytime", "failures", "absences"]]

predict = "G3"

X = np.array(data.drop([predict],1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

linear = linear_model.LinearRegression()

linear.fit(x_train, y_train)
accuracy = linear.score(x_test, y_test)
print(accuracy)

print('Coefficient: \n', linear.coef_)
print('Coefficient: \n', linear.intercept_)

