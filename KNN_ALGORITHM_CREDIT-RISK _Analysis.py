# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 15:36:45 2020

@author: admin
"""

import pandas as pd


cr=pd.read_csv("D:\\python\\csv/CreditRisk.csv")
cr
#LOAN STATUS IS APPROVED OR NO
#TARGET THE PEOPLE FOR A CAMPAGIN

cr.head()
cr.tail()
cr.describe()

#checking the NULL VALUE
cr.isnull().sum()

#MAX VALUE HAVING THE NULL

#0 becaus eassuming that bad
cr.Credit_History=cr.Credit_History.fillna(0)
#YES BECAUS 
cr.Self_Employed=cr.Self_Employed.fillna("Yes")
#
#REPLACING IT WITH THE MEDIAN
cr.Loan_Amount_Term.median()
cr.Loan_Amount_Term=cr.Loan_Amount_Term.fillna(cr.Loan_Amount_Term.median())

#DEPENdents
cr.Dependents=cr.Dependents.fillna(0)
#GENDER
cr.Gender=cr.Gender.fillna("Male")
#MARRIED 
cr.Married=cr.Married.fillna("No")
#LOAN AMMOUT
cr.LoanAmount=cr.LoanAmount.fillna(cr.LoanAmount.median())

#checking the NULL VALUE
cr.isnull().sum()

#-----------------------------------------------------------------
#CONVERTING NON NUMERIC TO NUMERIC

cr.Gender.replace({"Male":1,"Female":0},inplace=True)
cr.Married.replace({"Yes":1,"No":0},inplace=True)
cr.Self_Employed.replace({"Yes":1,"No":0},inplace=True)
cr.Education.replace({"Graduate":1,"Not Graduate":0},inplace=True)
cr.Property_Area.replace({"Semiurban":1,"Urban":2,"Rural":3},inplace=True)
cr.Loan_Status.replace({"Y":1,"N":0},inplace=True)

cr
#----------------------------------------------------------------
#dividing the data into x and y
cr_x=cr.iloc[:,1:12]#x
cr_y=cr.iloc[:,-1]#y[12 or -1 ]its okk Target variable

#-------------------------------------------------
import sklearn
from sklearn.model_selection import train_test_split

cr_x_train,cr_x_test,cr_y_train,cr_y_test=train_test_split(cr_x,cr_y,test_size=0.2,random_state=1111)

from sklearn.metrics import confusion_matrix,classification_report

    
from sklearn.neighbors import KNeighborsClassifier
lis=[]
for i in range(1,50):
    KNNmodel=KNeighborsClassifier(n_neighbors=i)
    KNNmodel.fit( cr_x_train, cr_y_train)
    pred_value=KNNmodel.predict(cr_x_test)
    pred_value
    table=confusion_matrix(pred_value,cr_y_test)
    b=table.diagonal().sum()/table.sum()*100
    #lis.append("a")
    lis.append(b)

lis

#plotting


import matplotlib.pyplot as plt
plt.plot(lis)
b