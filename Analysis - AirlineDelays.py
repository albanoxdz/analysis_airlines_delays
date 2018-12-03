# -*- coding: utf-8 -*-
"""
Spyder Editor

Este é um arquivo de script temporário.
"""
#Fonte:
#https://www.kaggle.com/timolee/a-home-for-pandas-and-sklearn-beginner-how-tos

import pandas as pd
import numpy as np

#prep
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV

#models
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, LinearRegression, Ridge, RidgeCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

#validation libraries
from sklearn.cross_validation import KFold, StratifiedKFold
from IPython.display import display
from sklearn import metrics

df = pd.read_csv("./DelayedFlights.csv")

days = {1:"Mon",
       2:"Tues",
       3:"Wed",
       4:"Thur",
       5:"Fri",
       6:"Sat",
       7:"Sun"}

df["DayOfWeek_name"] = df["DayOfWeek"].apply(lambda x: days[x])

df_head = df.head()

df = df.drop("Unnamed: 0",1)

df_head = df.head()

#Duplicating the dataset
#df_delays = df.drop(["Year","DepTime","CRSDepTime","ArrTime","CRSArrTime","UniqueCarrier","FlightNum","TailNum","ActulElapsedTime"])
df_delays = pd.DataFrame(df["Month"])
df_delays = df["Month","DayofMonth","DayOfWeek","WeatherDelay","NASDelay","SecurityDelay","LateAircraftDelay"]

#Adding a date column to the dataset
df_delays["DepartDate"] = pd.to_datetime(df_delays["Year"]*10000 + df_delays["Month"]*100 + df_delays["DayofMonth"],format='%Y%m%d')
#df_delays["DepartDate"] = pd.to_datetime(df_delays["Year"]*100000000 + df_delays["Month"]*1000000 + df_delays["DayofMonth"]*10000 + df_delays["DepTime"],format='%Y%m%d%f')
df_delays_head = df_delays.head()

############################################
#Understanding basic features of the dataset
############################################
airlines = df_delays['UniqueCarrier'].unique()
print(len(airlines))

tail_numbers = df_delays['TailNum'].unique()
print(len(tail_numbers))

airports_origin = df_delays['Origin'].unique()
print(len(airports_origin))

airports_dest = df_delays['Dest'].unique()
print(len(airports_dest))

diff_airports = np.setxor1d(airports_origin,airports_dest)
print(diff_airports)

#Visualize some information
import seaborn as sns

sns.countplot(df['UniqueCarrier'],label="Count")
sns.countplot(df['TailNum'],label="Count")
sns.countplot(df['Origin'],label="Count")
sns.countplot(df['Dest'],label="Count")

#Converting airlines column to category type
df_delays.dtypes
df_delays['UniqueCarrier'] = df_delays['UniqueCarrier'].astype('category')
df_delays['UniqueCarrier_cat'] = df_delays['UniqueCarrier'].cat.codes

df_delays['TailNum'] = df_delays['TailNum'].astype('category')
df_delays['TailNum_cat'] = df_delays['TailNum'].cat.codes

df_delays['Airport'] = df_delays['Origin'].astype('category')
df_delays['Airport_cat'] = df_delays['Airport'].cat.codes

df_delays_head = df_delays.head()

df_delays.describe()

#Split train and test datasets
delay_type = "CarrierDelay"
y = df_delays[delay_type]
featured_cols = [col for col in df_delays.columns if delay_type not in col]
X = df_delays[featured_cols]
X_train, X_valid, y_train, y_valid = train_test_split(X,y, test_size=0.2)

################################################
#Modeling
################################################

#Linear Model
lm = LinearRegression()
lm.fit(X_train,y_train)



