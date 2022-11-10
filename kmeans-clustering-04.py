### Importing the data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
#from scipy.spatial.distance import cdist

#### Reading the data telecom_churn for kmeans clustering
telecom=pd.read_excel(r"E:\DESKTOPFILES\suraj\assigments\k-means clusteringgg\Datasets_Kmeans\Telco_customer_churn (1).xlsx")

telecom.isna().sum() # check for count of NA'sin each column

telecom.isnull().sum()

telecom.describe() # To know the stats about the data

telecom.dtypes ## To know the data type of each variable in the telecom dataset 

### Dropping the some columns which are unnecessary
updatedTelecom=telecom.drop(columns=["Count","Quarter","Referred a Friend","Number of Referrals","Tenure in Months","Phone Service","Multiple Lines","Internet Service","Internet Type","Avg Monthly GB Download","Online Security","Online Backup","Device Protection Plan","Premium Tech Support","Streaming TV","Streaming Movies","Streaming Music","Unlimited Data","Contract"],axis=1)

### getting dummy variables 
listofcol=["Offer","Paperless Billing","Payment Method"]
dummyTelecomData = pd.get_dummies(data= updatedTelecom,columns = listofcol)

### Normalization function - Custom Function
# Range converts to: 0 to 1
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return(x)

df_norm=norm_func(dummyTelecomData.iloc[:,1:])

###### scree plot or elbow curve ############
TWSS=[]
k=list(range(2,18))

for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS

##scree plot
plt.plot(k,TWSS,"ms-");plt.xlabel("No_of_clusters");plt.ylabel("total_within_SS")

# Selecting 4 clusters from the above scree plot which is the optimum number of clusters 
model=KMeans(n_clusters=4)

model.fit(df_norm)

model.labels_ # getting the labels of clusters assigned to each row 

mb=pd.Series(model.labels_)  # converting numpy array into pandas series object 

dummyTelecomData["clust"]=mb  # creating a  new column and assigning it to new column 

telecom.head()

df_norm.head()

updatedTelecom.shape

dummyTelecomData.shape

###  Aggregating  the each cluster by the mean
dummyTelecomData.iloc[:,[19,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]]

Telecomdata1=dummyTelecomData.iloc[:,[19,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]]

Telecomdata1.iloc[:, 0:].groupby(Telecomdata1.clust).mean()

Telecomdata1.to_csv("Kmeans_TELECOM-CHURN.csv", encoding = "utf-8")

import os
 
os.getcwd()


Insights driven from the Telecomdata after the kmeans clustering

cluster-00 = Highest Totol Revenue = 11979.34  
             Lowest Total Revenue = 23.45   
             Highest Total Refunds =  49.57 as 2 times 
             Lowest Total Refunds = 0
             Highest Total Charges = 8684.8
             Lowest Total Charges = 19.25
             Highest Monthly Charges = 118.75
             Lowest Monthly Charges = 19.05
             Highest Total extradata Charges = 150  as 20 times 
             Lowest Total extradata Charges = 0
             
cluster-01 = Highest Totol Revenue = 11501.82
             Lowest Total Revenue = 21.40
             Highest Total Refunds =  49.79
             Lowest Total Refunds = 0
             Highest Total Charges =  8670.10
             Lowest Total Charges = 18.8
             Highest Monthly Charges = 118.60
             Lowest Monthly Charges = 18.25
             Highest Total extradata Charges =  150 as 9 times  
             Lowest Total extradata Charges =  0
             
cluster-02 = Highest Totol Revenue = 11272.18
             Lowest Total Revenue = 22.12
             Highest Total Refunds =  49.23
             Lowest Total Refunds = 0
             Highest Total Charges = 8349.7
             Lowest Total Charges = 18.85
             Highest Monthly Charges = 117.45
             Lowest Monthly Charges = 18.4
             Highest Total extradata Charges = 150 as  5 times
             Lowest Total extradata Charges = 0
             
             
cluster-03 = Highest Totol Revenue = 11795.78
             Lowest Total Revenue = 21.36
             Highest Total Refunds = 48.23
             Lowest Total Refunds = 0
             Highest Total Charges = 8594.4
             Lowest Total Charges = 18.85
             Highest Monthly Charges =18.55
             Lowest Monthly Charges = 117.35
             Highest Total extradata Charges =  150 as 8 times 
             Lowest Total extradata Charges = 0 
             
As compare to all 4 clusters 
                     Highest Totol Revenue = 11979.34
                     Lowest Total Revenue = 21.36
                     Highest Total Refunds = 49.79
                     Lowest Total Refunds = 0
                     Highest Total Charges = 8684.8
                     Lowest Total Charges = 18.8
                     Highest Monthly Charges = 118.75
                     Lowest Monthly Charges = 18.25
                     Highest Total extradata Charges = 150 as 42 times 
                     Lowest Total extradata Charges = 0
                          
                          