import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
## from scipy.spatial.distance import cdist

data=pd.read_excel(r"E:\DESKTOPFILES\suraj\assigments\k-means clusteringgg\Datasets_Kmeans\EastWestAirlines (1).xlsx")

data.describe()

data.shape

data.head()

data.dtypes


duplicate=data.duplicated()
duplicate
sum(duplicate)

data.isna().sum()

data.drop(['ID#'],axis=1,inplace=True)

data.rename(columns={'Award?':'Awards'},inplace=True)

######## Normalizing the data sets ####
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return(x)

# Normalized data frame (considering the numerical part of data)

df_norm=norm_func(data.iloc[:,1:])

###### scree plot or elbow curve ############

TWSS=[]
k=list(range(2,15))
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS

 # Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 3 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 3)
model.fit(df_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
data['clust'] = mb # creating a  new column and assigning it to new column 

data.head()
df_norm.head()

data.shape

###  Aggregating  the each cluster by the mean
data1 = data.iloc[:,[11,0,1,2,3,4,5,6,7,8,9,10]]
data1.head

data1.iloc[:, 1:].groupby(data1.clust).mean()
data1.to_csv("Kmeans_AIRLINES.csv", encoding = "utf-8")

import os
os.getcwd()


Insights from the airlines after kmeans clustering

CLuster-0= from the above airlines that the cluster-zero have highest balance of 456509 and lowest balance of 50,
            and Qual_miles have highest miles of 10074 and lowest Qual_miles have 0
            and days_since_enroll have highest days of 8296 days and  lowest days_since_enroll have 2 days 
        
CLuster-1= from the above airlines that the cluster-One have highest balance of 1125076 and lowest balance of 18497,
            and Qual_miles have highest miles of 6286 and lowest Qual_miles have 0
            and days_since_enroll have highest days of 8296 days and  lowest days_since_enroll have 450 days .
            
CLuster-2= from the above airlines that the cluster-Two have highest balance of 1704838 and lowest balance of 0,
            and Qual_miles have highest miles of 11148 and lowest Qual_miles have 0
            and days_since_enroll have highest days of 8296 days and  lowest days_since_enroll have 301 days .