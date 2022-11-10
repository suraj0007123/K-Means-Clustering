### Importing the data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
#from scipy.spatial.distance import cdist

### Reading the autoinsurance dataset into the python for kmeans clustering
autoinsurance=pd.read_csv(r"E:\DESKTOPFILES\suraj\assigments\k-means clusteringgg\Datasets_Kmeans\AutoInsurance (1).csv")


autoinsurance.info() ### To know the info about the dataframe

autoinsurance.describe() ## To know the stats about the dataframe

autoinsurance.dtypes ### To know the datatype of each variable

### Identify duplicates records in the data ###
duplicate=autoinsurance.duplicated()
duplicate
sum(duplicate)
autoinsurance.shape

#take categorical data into one file for lable encoding
insurancedata=autoinsurance[['Coverage','EmploymentStatus','Location Code','Policy Type','Policy','Renew Offer Type']]
insurancedata.columns

#######################
# Label Encoder
from sklearn.preprocessing import LabelEncoder 
# creating instance of labelencoder

labelencoder=LabelEncoder()

# Data Split into Input and Output variables
X=insurancedata.iloc[:,0:6]

y=insurancedata['Renew Offer Type']

insurancedata.columns

X['Coverage']=labelencoder.fit_transform(X['Coverage'])
X['EMplaymentStatus']=labelencoder.fit_transform(X['EmploymentStatus'])
X['Location Code']=labelencoder.fit_transform(X['Location Code'])
X['Policy Type']=labelencoder.fit_transform(X['Policy Type'])
X['Policy']=labelencoder.fit_transform(X["Policy"])
X['Renew Offer Type']=labelencoder.fit_transform(X['Renew Offer Type'])

### label encode y ###
y=labelencoder.fit_transform(y)
y=pd.DataFrame(y)

### we have to convert y to data frame so that we can use concatenate function
# concatenate X and y
insurance_new=pd.concat([X,y],axis=1)
insurance_new.columns

## rename column name
insurance_new=insurance_new.rename(columns={0:"Renew Offer Type"})

#take numerical data and renaming it.
autoinsurance.rename(columns={"Effective To Date":"effectivetodate","Customer Lifetime Value":"customerlifetimevalue","Monthly Premium Auto":"monthlypremiumauto","Months Since Last Claim":"monthssincelastclaim","Months Since Policy Inception":"monthssincepolicyinception","Number of Open Complaints":"numberofopencomplaints","Number of Policies":"numberofpolicies","Total Claim Amount":"totalclaimamount"},inplace=True)

#take numerical data for normalization
insurance1=autoinsurance[["customerlifetimevalue","Income","monthlypremiumauto","monthssincelastclaim","monthssincepolicyinception","numberofpolicies","totalclaimamount"]]

insurance1.describe() ## To know the stats of all the numerical columns

insurance1.info() # To know the info about the dataframe

autoinsurance.columns # To know the column names of the autoinsurance dataset

###Dropping the columns which are irrelevent in the autoinsurance datasets
insurancedata1=autoinsurance.drop(['Customer','effectivetodate'],axis=1)
insurancedata1.columns

# Normalization function 
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return(x)

# Normalized data frame (considering the numerical part of data)
df_norm=norm_func(insurance1.iloc[:,1:])

df_norm.describe()

df_norm.columns

###### scree plot or elbow curve ############
TWSS = []
k = list(range(2, 14))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)
TWSS

# Scree plot 
plt.plot(k, TWSS, 'bs-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 4)
model.fit(df_norm)

model.labels_ # getting the labels of clusters assigned to each row 

mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 

insurancedata1['clust'] = mb # creating a  new column and assigning it to new column 

insurancedata1.head()

df_norm.head()

insurancedata1.columns

insurancedata1.shape

insurancedata1 = insurancedata1.iloc[:,[22,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]]

insurancedata1.head()

###  Aggregating  the each cluster by the mean
insurancedata1.iloc[:,].groupby(insurancedata1.clust).mean()

insurancedata1.to_csv("Kmeans_AUTOINSURANCE.csv", encoding = "utf-8")

import os

os.getcwd()


Insights driven from autoinsurance after the kmeans clustering 

cluster-00= Hightest TotalClaim Amount = 2452.894264 
            Lowest Totalclaim amount = 0.3821
            Highest Number of policies = 7
            Lowest Number of policies = 1 as 1102 times 
            Highest Number of open complaints = 5 as 18 times 
            Lowest Number of open complaints = 0
            Highest Income = 99981 
            Lowest Income = 39815
            
            
cluster-01= Hightest TotalClaim Amount = 2893.239678
            Lowest Totalclaim amount = 0.517753
            Highest Number of policies = 6 as 7 times 
            Lowest Number of policies = 1 as 1145 times 
            Highest Number of open complaints = 5 as 14 times 
            Lowest Number of open complaints = 0
            Highest Income = 51859
            Lowest Income = 0
            
            
cluster-02= Hightest TotalClaim Amount = 2759.794354
            Lowest Totalclaim amount = 1.332349
            Highest Number of policies = 6 as 6 times 
            Lowest Number of policies = 1 as 1004 times 
            Highest Number of open complaints = 5 as 4 times 
            Lowest Number of open complaints = 0
            Highest Income = 91761
            Lowest Income = 0
            
cluster-03= Hightest TotalClaim Amount = 2059.2
            Lowest Totalclaim amount = 0.099007
            Highest Number of policies = 9 as 416 times 
            Lowest Number of policies = 4 
            Highest Number of open complaints = 5 as 10 times 
            Lowest Number of open complaints = 0
            Highest Income = 99845 as 6 times  
            Lowest Income = 0
            
As compare to all 4 clusters 
            Hightest TotalClaim Amount = 2893.24
            Lowest Totalclaim amount = 0.099007
            Highest Number of policies = 9 as 416 times
            Lowest Number of policies = 1 as 3251 times 
            Highest Number of open complaints = 5 as 56 times 
            Lowest Number of open complaints = 0
            Highest Income = 99981 
            Lowest Income = 0
                         
            