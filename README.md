# Ex-06-Feature-Transformation
# Aim:
1.To read and perform feature transformation for the given dataset.

# Explanation:
Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column (feature) and transform the values, which are useful for our further analysis. It is a technique by which we can boost our model performance.

# Algorithm:
## STEP 1
Read the given Data

## STEP 2
Clean the Data Set using Data Cleaning Process

## STEP 3
Apply Feature Transformation techniques to all the features of the data set

## STEP 4
Save the data to the file
# Program:
```
Name : Priyadharshini P
Register numnber : 212222100039

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats

df = pd.read_csv("/content/Data_to_Transform.csv")
df

df.head()

df.isnull().sum()

df.info()

df.describe()

df1 = df.copy()

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df['Highly Negative Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df['Moderate Negative Skew'],fit=True,line='45')
plt.show()

df['Highly Positive Skew'] = np.log(df['Highly Positive Skew'])

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

df['Moderate Positive Skew'] = np.log(df['Moderate Positive Skew'])

sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.show()

df['Highly Positive Skew'] = 1/df['Highly Positive Skew']

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

df['Highly Positive Skew'] = df['Highly Positive Skew']**(1/1.2)

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

df['Moderate Positive Skew_1'], parameters=stats.yeojohnson(df['Moderate Positive Skew'])

sm.qqplot(df['Moderate Positive Skew_1'],fit=True,line='45')
plt.show()

from sklearn.preprocessing import PowerTransformer
transformer=PowerTransformer("yeo-johnson")
df['ModerateNegativeSkew_2']=pd.DataFrame(transformer.fit_transform(df[['Moderate Negative Skew']]))
sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.show()

from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution = 'normal')
df['ModerateNegativeSkew_2'] = pd.DataFrame(qt.fit_transform(df[['Moderate Negative Skew']]))
sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.show()
```
# OUTPUT:
# Dataset:
![image](https://user-images.githubusercontent.com/119558093/234964150-a9120c2e-aa20-455e-9193-b24e7d9b6777.png)

# Head:
![image](https://user-images.githubusercontent.com/119558093/234964258-f1cf6d3e-ba52-4119-8eac-c4ce2f022dfa.png)

# Null data:
![image](https://user-images.githubusercontent.com/119558093/234964354-632a3336-7f40-4ed3-9a49-d9978fc9d071.png)

# Information:
![image](https://user-images.githubusercontent.com/119558093/234964385-d1440cf2-d398-4e8c-8689-e34091fb175d.png)

# Description:
![image](https://user-images.githubusercontent.com/119558093/234964473-87b5f83a-3f0a-4af5-8c03-23e0b3faf871.png)

# Highly Positive Skew:

![image](https://user-images.githubusercontent.com/119558093/234964581-29bb0ddb-33a2-4d68-b5d5-dbfea43504a2.png)
# Highly Negative Skew:
![image](https://user-images.githubusercontent.com/119558093/234964667-6aa9e162-3e74-495c-ba90-a7385ac09738.png)
# Moderate Positive Skew:
![image](https://user-images.githubusercontent.com/119558093/234964798-e1a4f668-700d-4053-ab96-2b0391e2e69d.png)
# Moderate Negative Skew:
![image](https://user-images.githubusercontent.com/119558093/234964998-753edc39-825a-496b-a724-51e3b4ea5c0e.png)
# Log of Highly Positive Skew:
![image](https://user-images.githubusercontent.com/119558093/234965059-b782f98b-7134-4263-a84e-4901ff60bd5b.png)

# Log of Moderate Positive Skew:
![image](https://user-images.githubusercontent.com/119558093/234965143-215ab7f1-ac9b-4b91-a090-a792f52bd4df.png)

# Reciprocal of Highly Positive Skew:
![image](https://user-images.githubusercontent.com/119558093/234965212-ba8d0442-0c07-42eb-a951-393cd760e55a.png)
# Square root tranformation:
![image](https://user-images.githubusercontent.com/119558093/234965343-a22a9435-ca0b-4f7e-88ab-a9d0a50eb5bc.png)

# Power transformation of Moderate Positive Skew:
![image](https://user-images.githubusercontent.com/119558093/234965427-8a98d977-7cb1-4ab2-97c2-aaeeefea3e33.png)

# Power transformation of Moderate Negative Skew:
![image](https://user-images.githubusercontent.com/119558093/234965500-eb53b723-aecd-474b-bdb4-35c5abba0a60.png)

# Quantile transformation:
![image](https://user-images.githubusercontent.com/119558093/234965595-c440bb28-78bc-49d6-b1c2-d9b78a782c6c.png)

# Result :
Thus, Feature transformation is performed and executed successfully for the given dataset.





