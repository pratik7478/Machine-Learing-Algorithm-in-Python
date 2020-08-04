# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 12:11:27 2020

@author: admin
"""

import pandas as pd


cr=pd.read_csv("D:\\python\\csv/BreastCancer.csv")
cr
#LOAN STATUS IS APPROVED OR NO
#TARGET THE PEOPLE FOR A CAMPAGIN

cr.head()
cr.tail()
cr.describe()

#checking the NULL VALUE
cr.isnull().sum()


cr=cr.rename(columns={'Bare.nuclei': 'bare_nuclei', 'Cl.thickness': 'cl_thickness'})
cr
#MAX VALUE HAVING THE NULL

#0 becaus eassuming that bad
cr.bare_nuclei=cr.bare_nuclei.fillna(cr.bare_nuclei.mean())

#cr.bare_nuclei=cr.bare_nuclei.fillna(1)

cr.isnull().sum()

cr
#-----------------------------------------------------------------
#CONVERTING NON NUMERIC TO NUMERIC

cr.Class.replace({"benign":1,"malignant":0},inplace=True)

cr
#----------------------------------------------------------------
#dividing the data into x and y
cr_x=cr.iloc[:,2:11]#x
cr_y=cr.iloc[:,-1]#y[12 or -1 ]its okk Target variable

#-------------------------------------------------
import sklearn
from sklearn.model_selection import train_test_split

cr_x_train,cr_x_test,cr_y_train,cr_y_test=train_test_split(cr_x,cr_y,test_size=0.2,random_state=2222)

from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()
logmodel.fit( cr_x_train, cr_y_train)

#BUILDING THE MODLE ONLY X_TEST
#PREDICTION
pred_value=logmodel.predict(cr_x_test)
pred_value


#BUILDING THE CONFUSION MATRIX Y_TEST IS REQUIRED

from sklearn.metrics import confusion_matrix

table=confusion_matrix(pred_value,cr_y_test)
table

table.diagonal().sum()/table.sum()*100
