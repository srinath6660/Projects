import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_data = pd.read_excel('Data_Train.xlsx')
test_data = pd.read_excel('Test_set.xlsx')
dataset = train_data.append(test_data, sort=False)

dataset['Date'] = dataset['Date_of_Journey'].str.split('/').str[0]
dataset['Month'] = dataset['Date_of_Journey'].str.split('/').str[1]
dataset['Year'] = dataset['Date_of_Journey'].str.split('/').str[2]
dataset = dataset.drop(['Date_of_Journey'],axis = 1)

dataset['Date'] = dataset['Date'].astype(int)
dataset['Year'] = dataset['Year'].astype(int)
dataset['Month'] = dataset['Month'].astype(int)

dataset.dtypes

dataset.isnull().sum()

dataset['Total_Stops'] = dataset['Total_Stops'].fillna('1 stop')
dataset['Total_Stops'] = dataset['Total_Stops'].replace('non-stop', '0 stop')

dataset['Stops'] = dataset['Total_Stops'].str.split(' ').str[0]
dataset = dataset.drop(['Total_Stops'], axis = 1)
dataset['Stops'] = dataset['Stops'].astype(int)

dataset['Additional_Info'].value_counts()

dataset['Arrival_Time']=dataset['Arrival_Time'].str.split(' ').str[0]

dataset['Arrival_hour'] = dataset['Arrival_Time'].str.split(':').str[0]
dataset['Arrival_mintute'] = dataset['Arrival_Time'].str.split(':').str[1]
dataset = dataset.drop(['Arrival_Time'], axis = 1)
dataset['Arrival_hour'] = dataset['Arrival_hour'].astype(int)
dataset['Arrival_mintute'] = dataset['Arrival_mintute'].astype(int)

dataset.head()

dataset['Dep_hour'] = dataset['Dep_Time'].str.split(':').str[0]
dataset['Dep_min'] = dataset['Dep_Time'].str.split(':').str[1]
dataset['Dep_hour'] = dataset['Dep_hour'].astype(int)
dataset['Dep_min'] = dataset['Dep_min'].astype(int)
dataset = dataset.drop(['Dep_Time'], axis = 1)

dataset.isnull().sum()
dataset = dataset.drop(['Duration'], axis = 1)

dataset['Route_1']=dataset['Route'].str.split('→ ').str[0]
dataset['Route_2']=dataset['Route'].str.split('→ ').str[1]
dataset['Route_3']=dataset['Route'].str.split('→ ').str[2]
dataset['Route_4']=dataset['Route'].str.split('→ ').str[3]
dataset['Route_5']=dataset['Route'].str.split('→ ').str[4]

dataset['Route_1'].fillna("None",inplace=True)
dataset['Route_2'].fillna("None",inplace=True)
dataset['Route_3'].fillna("None",inplace=True)
dataset['Route_4'].fillna("None",inplace=True)
dataset['Route_5'].fillna("None",inplace=True)
dataset = dataset.drop(['Route'], axis = 1)
dataset.isnull().sum()

dataset['Price'].fillna((dataset['Price'].mean()),inplace=True)
dataset.isnull().sum()
dataset.dtypes

from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
for col in dataset.columns:
    if dataset[col].dtypes == 'object':
        dataset[col] = enc.fit_transform(dataset[col])

X = dataset.drop(['Price'], axis = 1)
y = dataset['Price']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

from sklearn.ensemble import RandomForestRegressor
reg = RandomForestRegressor(random_state = 0)
reg.fit(X_train, y_train)

pred = reg.predict(X_test)

plt.scatter(y_test,pred)















