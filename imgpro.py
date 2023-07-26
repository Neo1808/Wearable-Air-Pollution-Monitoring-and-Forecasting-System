import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle
import joblib
dataset = pd.read_csv('Data Set.csv')


X = dataset.iloc[:, [4,5]].values
y = dataset.iloc[:, 6].values


from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.25, random_state=0)

from sklearn.preprocessing import StandardScaler


sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)


from sklearn.svm import SVC


classifier = SVC(kernel='linear')


classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
print(y_pred)
from sklearn.metrics import accuracy_score


print(accuracy_score(y_test, y_pred))

joblib.dump(classifier, 'mq.pkl')