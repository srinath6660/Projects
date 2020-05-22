import numpy as np
import pandas as pd
data = pd.read_csv('SMSSpamCollection.tsv', sep = '\t')
data = data.rename(columns = {'v1':'label', 'v2':'text'})
data.describe()
data['label'].value_counts()
#data['spam'] = data['label'].map({'spam':1, 'ham':0}).astype(int)
data['length'] = data['text'].apply(len)
data.isnull().sum()

from sklearn.model_selection import train_test_split
X = data['text']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X_train_counts = cv.fit_transform(X_train)
X_train_counts.shape

from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer()
X_train_tfidf = tfidf.fit_transform(X_train_counts)
X_train_tfidf.shape

from sklearn.feature_extraction.text import TfidfVectorizer  # This process combines both count vectorization and tfidf transformation
tfidfvec = TfidfVectorizer()
X_train_tfidfvec = tfidfvec.fit_transform(X_train)


from sklearn.svm import LinearSVC
clf = LinearSVC()
clf.fit(X_train_tfidfvec, y_train)

from sklearn.pipeline import Pipeline  # Does everything in one line of code(count vectorization, tfidf tranformation(tfidf vectorization)+fit to the training model)
text_clf = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LinearSVC())])
text_clf.fit(X_train, y_train)

pred = text_clf.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, pred)
cr = classification_report(y_test, pred)
acc = accuracy_score(y_test, pred)

# Predicting new msg
text_clf.predict(['Hi how are you doing today?'])
text_clf.predict(['Congratulations! you won $10000'])
































