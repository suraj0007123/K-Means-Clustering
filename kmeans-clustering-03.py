## Importing the data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
#from scipy.spatial.distance import cdist

######## Reading the dataset insurance into the python for Kmeans clustering
insurance=pd.read_csv(r"E:\DESKTOPFILES\suraj\assigments\k-means clusteringgg\Datasets_Kmeans\Insurance Dataset.csv")

insurance.info() ## To know the info about the data

insurance.shape ## To know the shape of the dataset 

insurance.dtypes ### To know the data-type of each variable in the insurance dataset

insurance.isnull().sum() #### check for count of NA'sin each column

### Identify duplicates records in the data ###
duplicate=insurance.duplicated()
duplicate
sum(duplicate)

### Normalization function - Custom Function
# Range converts to: 0 to 1
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return(x)

df_norm=norm_func(insurance.iloc[:,0:])

df_norm.describe()

###### scree plot or elbow curve ############
TWSS=[]
k=list(range(2,12))

for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)
TWSS

# Scree plot 
plt.plot(k,TWSS,"cs-");plt.xlabel("No_of_clusters");plt.ylabel("totol_within_SS")

# Selecting 4 clusters from the above scree plot which is the optimum number of clusters 
model=KMeans(n_clusters=4) 
model.fit(df_norm)

model.labels_  # getting the labels of clusters assigned to each row 

sb=pd.Series(model.labels_) # converting numpy array into pandas series object 

insurance["clust"]=sb # creating a  new column and assigning it to new column

insurance.head()

df_norm.head()

insurance1=insurance.iloc[:,[5,0,1,2,3,4]]

###  Aggregating  the each cluster by the mean
insurance1.iloc[:,0:].groupby(insurance1.clust).mean()

insurance.to_csv("KMeans_INSURANCE.csv",encoding="utf-8")

import os

os.getcwd()

The Insights Driven from the given insurance dataset 
cluster-0 = Highest Claims made = 13810.05025 
           lowest claims made =  5327.532
           Highest Premium-paid = 21750
           lowest Premium-paid = 8950
           Highest Days to Renew = 165 days as twice times 
           lowest Days to Renew = 1 day
           Highest Age = 69 years old
           lowest Age = 23 years as twice times 

cluster-1 = Highest Claims made = 99676.74 
           lowest claims made =  10986.18
           Highest Premium-paid = 29900
           lowest Premium-paid = 15250
           Highest Days to Renew = 321 days 
           lowest Days to Renew =  48  days
           Highest Age = 82 years old as twice times
           lowest Age = 50 years  

cluster-2 = Highest Claims made =  56927.91
           lowest claims made =  3890.076
           Highest Premium-paid = 13450
           lowest Premium-paid = 2800
           Highest Days to Renew = 321 days as twice times 
           lowest Days to Renew =  165 days
           Highest Age = 70 years old as twice times
           lowest Age =  26 years              

cluster-3 = Highest Claims made =  5406.818
           lowest claims made =  1978.261
           Highest Premium-paid = 9150
           lowest Premium-paid = 2950
           Highest Days to Renew = 144 days 
           lowest Days to Renew =  1 day
           Highest Age = 53 years 
           lowest Age = 23 years          
           
As compare to all 4 clusters 
            Highest Claims made =  99676.74419
            lowest claims made =  1978.26087
            Highest Premium-paid = 29900
            lowest Premium-paid = 2800
            Highest Days to Renew = 321 days 3 times   
            lowest Days to Renew =  1 day  as two times 
            Highest Age = 82 years as 2 times
            lowest Age = 23 years as 3 times         

           
