## Importing data
import pandas as pd
# Read data into Python
airdata=pd.read_excel(r"E:\DESKTOPFILES\suraj\assigments\k-means clusteringgg\Datasets_Kmeans\EastWestAirlines (1).xlsx")

airdata.describe() # To know the stats about the data

airdata.drop(['ID#'], axis=1, inplace=True)

### EDA
### Measures of Central Tendency / First moment business decision
airdata.mean()
airdata.median()
airdata.mode()

# pip install numpy
from scipy import stats
stats.mode(airdata)

# Measures of Dispersion / Second moment business decision
airdata.var() # Variance
airdata.std()  # Standard Deviation

range=airdata.iloc[:,0:].max()-airdata.iloc[:,0:].min()
range

# Third moment business decision
airdata.skew()

# Fourth moment business decision
airdata.kurt()


### Data Visualization
import numpy as np
airdata.shape

import matplotlib.pyplot as plt
plt.bar(height=airdata.Balance,x=np.arange(1,4000,1))
plt.hist(airdata.Balance)
plt.bar(height=airdata['Qual_miles'],x=np.arange(1,4000,1))

import seaborn as sns
import numpy as np

## Identifying the outliers with the help of boxplot 
sns.boxplot(airdata.Balance)

########## Winsorization ###############
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',
                        tail='both',
                        fold=1.5,
                        variables=['Balance'])

df_t=winsor.fit_transform(airdata[['Balance']])

sns.boxplot(df_t.Balance)

airdata.columns

sns.boxplot(airdata.Qual_miles)

from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',
                         tail='both',
                         fold=1.5,
                         variables=['Qual_miles'])

df_t=winsor.fit_transform(airdata[['Qual_miles']])

sns.boxplot(df_t.Qual_miles)

airdata.columns

sns.boxplot(airdata.cc1_miles)

sns.boxplot(airdata.cc2_miles)

from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',
                          tail='both',
                          fold=1.5,
                          variables=['cc2_miles'])

df_t=winsor.fit_transform(airdata[['cc2_miles']])

sns.boxplot(df_t.cc2_miles)

sns.boxplot(airdata.cc3_miles)

from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',
                          tail='both',
                          fold=1.5,
                          variables=['cc3_miles'])

df_t=winsor.fit_transform(airdata[['cc3_miles']])

sns.boxplot(df_t.cc3_miles)

sns.boxplot(airdata.Bonus_miles)

from feature_engine.outliers import Winsorizer

winsor=Winsorizer(capping_method='iqr',
                          tail='both',
                          fold=1.5,
                          variables=['Bonus_miles'])

df_t=winsor.fit_transform(airdata[['Bonus_miles']])

sns.boxplot(df_t.Bonus_miles)

sns.boxplot(airdata.Bonus_trans)

from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',
                          tail='both',
                          fold=1.5,
                          variables=['Bonus_trans'])

df_t=winsor.fit_transform(airdata[['Bonus_trans']])

sns.boxplot(df_t.Bonus_trans)

sns.boxplot(airdata.Flight_miles_12mo)

from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',
                          tail='both',
                          fold=1.5,
                          variables=['Flight_miles_12mo'])

df_t=winsor.fit_transform(airdata[['Flight_miles_12mo']])

sns.boxplot(df_t.Flight_miles_12mo)

sns.boxplot(airdata.Flight_trans_12)

from feature_engine.outliers import Winsorizer

winsor=Winsorizer(capping_method='iqr',
                          tail='both',
                          fold=1.5,
                          variables=['Flight_trans_12'])

df_t=winsor.fit_transform(airdata[['Flight_trans_12']])

sns.boxplot(df_t.Flight_trans_12)

sns.boxplot(airdata.Days_since_enroll)
### Changing the name of variable (Rename)
airdata.rename(columns={'Award?':'Award'},inplace=True)

sns.boxplot(airdata.Award)
 
### Identify duplicates records in the data ###
duplicate=airdata.duplicated()
duplicate
sum(duplicate)
data1 = airdata.drop_duplicates()

#### zero variance and near zero variance ######
airdatavar=airdata.var()
airdatavar
###### Therefore the variance for cc2_miles and cc3_miles nearer to zero, so we can eliminate that columns
airdata.drop(['cc2_miles','cc3_miles'],axis=1,inplace=True)
airdata

### Missing values Checking #######
nullcolumns=np.where(airdata.isna()==True)
airdata.isna().sum()

### Normalization function - Custom Function
# Range converts to: 0 to 1
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return(x)

# Normalized data frame (considering the numerical part of data)
df_norm=norm_func(airdata.iloc[:,1:])

df_norm.describe() 

airdata.shape

airdata.columns

from sklearn.cluster import	KMeans
# from scipy.spatial.distance import cdist 
