import numpy as np
import pandas as pd

dataset = pd.read_csv('BankNote_Authentication.csv')

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)

import pickle
pickle_out = open('MyClassifier.pkl', 'wb')
pickle.dump(classifier, pickle_out)
pickle_out.close()






















