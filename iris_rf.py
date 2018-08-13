from sklearn import linear_model
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
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

model= RandomForestClassifier()
model.fit(feature,label)
predicted= model.predict(X_test)
print(predicted)
print("Test set score: {:.2f}".format(np.mean(predicted == y_test['Species'].ravel())))
