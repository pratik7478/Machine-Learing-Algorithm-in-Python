# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 15:06:58 2020

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


from sklearn.ensemble import RandomForestClassifier
rfmodel=RandomForestClassifier()
#4erfmodel=RandomForestClassifier(n_estimators=100)
rfmodel.fit( cr_x_train, cr_y_train)


#BUILDING THE MODLE ONLY X_TEST
#PREDICTION
pred_value=rfmodel.predict(cr_x_test)
pred_value

"""
#a=logmodel.predict_proba(cr_x)
#type(a)
#a=pd.DataFrame(a)
#a
#a.rename(columns={a.columns[0]: "PROB0", a.columns[1]: "PROB1"},inplace=True)
#a
#a=pd.concat([a,cr.Loan_ID],axis=1)
#a.sort_values(['PROB1'],ascending=False)
"""

#BUILDING THE CONFUSION MATRIX Y_TEST IS REQUIRED
   
from sklearn.metrics import confusion_matrix,classification_report

table=confusion_matrix(pred_value,cr_y_test)
table

print(classification_report(cr_y_test,pred_value))


table.diagonal().sum()/table.sum()*100

cr.corr()
cr.shape()
#----------------------------------------------
a=rfmodel.feature_importances_
a
list(a)

#-------------------------------------------------------
#for GETTING TO KNOW WHICH VARIABLE IS HAVING THE HIGGEST VALUE

f_score=pd.DataFrame({"Importance":rfmodel.feature_importances_,"Variable_Name":cr_x_train.columns})
 
f_score.sort_values(['Importance'],ascending=False)

#11 number because OF THE 11 

#we can get the idea where the value is higgest 0.