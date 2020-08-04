# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 13:48:21 2020

@author: admin
""" 

import pandas as pd


df=pd.read_csv("D:\\python\\csv/snsdata.csv")
df

df.head()
df.tail()
df.describe()
df.shape

df.isnull().sum()

df.age=df.age.fillna(df.age.median())
df.gender.replace({"M":1,"F":0},inplace=True)

df.gender.value_counts()


df.gender=df.gender.fillna(0)


from sklearn.cluster import KMeans

df_KMeans=KMeans(n_clusters=3)
df_KMeans.fit(df)

df_KMeans.labels_
len(list(df_KMeans.labels_))
len(df)

#how to cee the centroid WHAT WOULD BE THE CEnTROID IF THE

df_KMeans.cluster_centers_ 
#gives the centroid for each cluster
#each cluster will have the 40 co ordinate as gthere are 40 columns in og data

cen=pd.DataFrame(df_KMeans.cluster_centers_)
cen

df_KMeans.fit(df).score(df)



#to find the value of K


nc=range(1,8)
Kmeans=[KMeans(n_clusters=i) for i in nc]
score=[Kmeans[i].fit(df).score(df) for i in range(len(Kmeans))]
print(score)


import matplotlib.pyplot as plt
plt.plot(nc,score)

import numpy as np
score=np.absolute(score)


plt.plot(nc,score,marker="*",color="r")
plt.xlabel("NUMER OF CLUSTER")
plt.ylabel("SUM OF SQUARED DISTANCE")
plt.title("ELBOW PLOT ON SNS DATA")


#no the value of k=4 so agaon building the final model
df_KMeans=KMeans(n_clusters=4)
df_KMeans.fit(df)

df_KMeans.labels_

df_KMeans.cluster_centers_ 
#gives the centroid for each cluster
#each cluster will have the 40 co ordinate as gthere are 40 columns in og data

cen=pd.DataFrame(df_KMeans.cluster_centers_)
cen

df_KMeans.fit(df).score(df)

#ARRAY VL NOT GET ATTACHED TO THE DATA FRAME SO WE HAVE TO ADD THE COLUMN OF THE LABEL

df_clust=pd.concat([df,pd.Series(df_KMeans.labels_)],axis=1)
df
df_clust.shape
df_clust.head()

df_clust.rename(columns={df_clust.columns[40]:"CLUSTER_NUMBER"},inplace=True)
df_cluster=df_clust.sort_values(['CLUSTER_NUMBER'])
df_cluster
