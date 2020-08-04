# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 16:07:22 2020

@author: admin
"""


import pandas as pd
import numpy as np

df=pd.read_csv("D:\\python\\csv/LungCapData.csv")
df

df.Smoke.replace({"no":0,"yes":1},inplace=True)
df.Gender.replace({"female":0,"male":1},inplace=True)
df.Caesarean.replace({"no":0,"yes":1},inplace=True)


df.head()
df.tail()
df.describe()

#checking the NULL VALUE
df.isnull().sum()

#
#dividing the data into x and y
df_x=df.iloc[:,1:6]#x
df_y=df.iloc[:,0]#y[12 or -1 ]its okk Target variable

#-------------------------------------------------
import sklearn
from sklearn.model_selection import train_test_split

df_x_train,df_x_test,df_y_train,df_y_test=train_test_split(df_x,df_y,test_size=0.2,random_state=1111)

from sklearn import linear_model
lmodel=linear_model.LinearRegression()
lmodel.fit(df_x_train,df_y_train)

pred_value=lmodel.predict(df_x_test)
pred_value

#GIVE US THE COEFFICIEFT
lmodel.coef_

#GIVE US THE INTERCEPT
lmodel.intercept_

#CORRESPONDING THE R-square
a=lmodel.score(df_x_train,df_y_train)
a
b=a*a
b
1-[(1-(0.7222)*(5))/6-5-1]
z