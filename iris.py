import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

def species_to_numeric(series):
    if series['Species'] == 'Iris-setosa':
        return 0
    elif series['Species'] == 'Iris-versicolor':
        return 1
    elif series['Species'] == 'Iris-virginica':
        return 2

data = pd.read_csv("Iris.csv")
#print(data.keys())

#print("Target names: {}".format(data['Species']))
feature = pd.DataFrame(data,columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])
#print(feature)
data['Species'] = data.apply(species_to_numeric, axis='columns')
label = pd.DataFrame(data, columns=['Species'])
#print(label)

X_train, X_test, y_train, y_test = train_test_split(feature, label, random_state=0)
#print("XTrain shape {}".format(X_train.shape))
#print("YTrain shape {}".format(y_train.shape))
#print("XTest shape {}".format(X_test.shape))
#print("YTest shape {}".format(y_test.shape))

#print(X_train)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,np.ravel(y_train,order='C'))

# making prediction here
X_new = np.array([[5, 2.9, 1, 0.2]])
print("X_new.shape: {}".format(X_new.shape))

prediction = knn.predict(X_new)
print("Prediction:{}".format(prediction))

#prits 0 her i.r setosa

#testing
y_pred = knn.predict(X_test)
print("Test set predictions:\n {}".format(y_pred))
#print(X_test)
#print(y_test)
print("Test set score: {:.2f}".format(np.mean(y_pred == y_test['Species'].ravel())))
# print(y_test['Species'].ravel())
