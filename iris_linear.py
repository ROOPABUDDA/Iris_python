from sklearn import linear_model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def species_to_numeric(series):
    if series['Species'] == 'Iris-setosa':
        return 0
    elif series['Species'] == 'Iris-versicolor':
        return 1
    elif series['Species'] == 'Iris-virginica':
        return 2

data = pd.read_csv("Iris.csv")
feature = pd.DataFrame(data,columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])
data['Species'] = data.apply(species_to_numeric, axis='columns')
label = pd.DataFrame(data, columns=['Species'])

X_train, X_test, y_train, y_test = train_test_split(feature, label, random_state=0)

#create lr object
linear = linear_model.LinearRegression()
#train model
linear.fit(X_train, y_train)
linear.score(X_train, y_train)

#coefficient , intercept
print('coefficient:\n ',linear.coef_)
print('intercept:\n ', linear.intercept_)

#predict ouput here

predicted = linear.predict(X_test)
#print(predicted)
#n = np.array([])
#for each in predicted:
#    np.append(n,int(round(each)))
#print(n)
#print(y_test)

arr = np.array([],int)
for each in predicted:
    arr = np.hstack((arr, np.array(int(round(each)))))
#print(arr)
#print(y_test)
print("Test set score: {:.2f}".format(np.mean(arr == y_test['Species'].ravel())))
