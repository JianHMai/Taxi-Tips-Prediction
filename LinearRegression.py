import pandas as pd
import numpy as np
import csv
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle
import matplotlib.pyplot as plt

data = pd.read_csv("TaxiTrip2021Subset.csv")

train = data.sample(frac=0.8)
test = data.drop(train.index).sample(frac=1.0)

trainX = train[['Trip Seconds', 'Trip Miles', 'Fare']]
trainY = train['Tips']

testX = test[['Trip Seconds', 'Trip Miles', 'Fare']]
testY = test['Tips']

reg = LinearRegression().fit(trainX, trainY)
predY = reg.predict(testX)
MSE = mean_squared_error(testY, predY)

plt.scatter(testX['Fare'], testY, s=5, color="blue", label="original")
plt.scatter(testX['Fare'], predY, lw=2, color="red", label="fitted")
plt.legend()
plt.show()

# Save model 
pickle.dump(reg, open('LinearRegressionModel.sav', 'wb'))