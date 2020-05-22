import numpy as np
import pandas as pd

dataset = pd.read_csv('car_evaluation.csv')
dataset.isnull().sum()



dataset['no_doors'].value_counts()
dataset['no_doors'].replace('5more', '5', inplace = True)
dataset['no_doors'].value_counts()

dataset['no_persons'].value_counts()
dataset['no_persons'].replace('more', '5', inplace = True)
dataset['no_persons'].value_counts()

dataset['no_doors'] = dataset['no_doors'].astype(int)
dataset['no_persons'] = dataset['no_persons'].astype(int)

dataset.dtypes

from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
for col in dataset.columns:
    if dataset[col].dtypes == 'object':
        dataset[col] = enc.fit_transform(dataset[col])

X = dataset.iloc[:, :6]
y = dataset.iloc[:, -1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 1, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
        
        
        
    