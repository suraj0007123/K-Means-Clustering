### Importing data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
### Reading the data
##### Kmeans on crime Data set 
crime=pd.read_csv(r"E:\DESKTOPFILES\suraj\assigments\k-means clusteringgg\Datasets_Kmeans\crime_data (1).csv")
# check for count of NA'sin each column
crime.isna().sum()

crime.isnull().sum()

### Identify duplicates records in the data ###
duplicate=crime.duplicated()
duplicate
sum(duplicate)

crime.head()

crime.describe()

crime.shape

crime.columns

### Renaming the states column which is unnamed in the dataset
crime1=crime.rename(columns={"Unnamed":"states"},inplace=True)

### import KMeans from sklearn library
from sklearn.cluster import KMeans
## from scipy.spatial.distance import cdist


# Normalization function 
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return(x)

df_norm = norm_func(crime.iloc[:, 1:])

df_norm.describe()

###### scree plot or elbow curve ############
TWSS = []
k = list(range(2,15))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 4 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 4)
model.fit(df_norm)

model.labels_ # getting the labels of clusters assigned to each row 

mo = pd.Series(model.labels_)  # converting numpy array into pandas series object 

crime['clust'] = mo # creating a  new column and assigning it to new column 

crime.head()
df_norm.head()

crime.shape

crime2 = crime.iloc[:,[5,0,1,2,3,4]]
crime2.head()

###  Aggregating  the each cluster by the mean
crime2.iloc[:, 2:6].groupby(crime2.clust).mean()

crime2.to_csv("Kmeans_CRIME.csv", encoding = "utf-8")

import os
os.getcwd()

insights driven from the crimedata after the kmeans clustering 

CLuster-0= from the above crime that the cluster-zero have 
           Highest murders = 17.4 cases in Georgia state
           lowest murders = 8.8 cases  in Arkanas state
           Highest Assault = 337 cases in North Caroline state(the meaning of assault=a concerted attempt to do something demanding.)
           lowest Assault = 188 cases  in Tennessee state
           Highest UrbanPop = 66 cases in Louisiana state
           lowest UrbanPop = 44 cases  in Mississippi state
           Highest Rape = 26.9 cases in Tennessee state
           lowest Rape = 16.1 cases  in North Carolina state
           
CLuster-1= from the above crime that the cluster-one have 
           Highest murders = 9.7 cases in Kentucky state
           lowest murders = 0.8 cases  in North Dakota state
           Highest Assault = 120 cases in Idaho state(the meaning of assault=a concerted attempt to do something demanding.)
           lowest Assault = 45 cases  in North Dakota state
           Highest UrbanPop = 66 cases in Minnesota,Wisconsin states
           lowest UrbanPop = 32 cases  in Vermont state
           Highest Rape = 16.5 cases in Nebraska state
           lowest Rape = 7.3 cases  in North Dakota state
             
CLuster-2 = from the above crime that the cluster-Two have 
           Highest murders = 9 cases in Missouri state
           lowest murders = 3.2 cases  in Utah state
           Highest Assault = 238 cases in Delaware state(the meaning of assault=a concerted attempt to do something demanding.)
           lowest Assault = 46 cases  in Hawaii state
           Highest UrbanPop = 89 cases in New Jersey states
           lowest UrbanPop = 60 cases  in Wyoming state
           Highest Rape = 29.3 cases in Oregan state
           lowest Rape = 8.3 cases  in Rhode Island state
                      
           
CLuster-3 = from the above crime that the cluster-Three have 
           Highest murders = 15.4 cases in Florida state
           lowest murders = 7.9 cases  in Colorado state
           Highest Assault = 335 cases in Florida state(the meaning of assault=a concerted attempt to do something demanding.)
           lowest Assault = 201 cases  in Texas state
           Highest UrbanPop = 91 cases in California states
           lowest UrbanPop = 48 cases  in Alaska state
           Highest Rape = 46 cases in Nevada state
           lowest Rape = 24 cases  in Illnois Island state
           
As Compare to all the clusters that 4 clusters
           Highest murders = 17.4 cases in Georgia State
           lowest murders = 0.8 cases  in North Dakota state
           Highest Assault = 337 cases in North carolina state(the meaning of assault=a concerted attempt to do something demanding.)
           lowest Assault = 45 cases  in North Dakota state
           Highest UrbanPop = 91 cases in California states
           lowest UrbanPop = 32 cases  in Vermont state
           Highest Rape = 46 cases in Nevada state
           lowest Rape = 7.3 cases  in North Dakota Island state
