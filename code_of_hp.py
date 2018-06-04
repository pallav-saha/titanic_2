# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd

hp=pd.read_csv("hp_price.csv")
hp.columns
hp.drop(["total_floors","builder_name","floor"],axis=1,inplace=True)
hp.isnull().sum()
hp.Cand=hp.Cand.fillna("Ragu")

#outlier

hp.boxplot(column="bhk")
hp=hp[hp["bhk"]<4]

'''hp.boxplot(column="age")
hp=hp[hp["age"]<3]
hp=hp[hp["age"]>3]'''


hp.boxplot(column="sqft")
hp=hp[hp["sqft"]<2400]

hp.boxplot(column="price")
hp=hp[hp["price"]<25000000]

hp=pd.get_dummies(hp,drop_first=True)

X=hp.drop("price",axis=1)
y=hp.price

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=100)
regressor=LinearRegression()
regressor.fit(X_train,y_train)
y_pred=regressor.predict(X_test)
r2_score(y_test,y_pred)


