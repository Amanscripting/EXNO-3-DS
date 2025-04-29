## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd

df=pd.read_csv("/content/Encoding Data (2).csv")

df
```
![Screenshot 2025-04-18 131409](https://github.com/user-attachments/assets/38a13d16-2601-4abb-b8bd-3fc024a46635)
```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder

pm=['Hot','Warm','Cold']

e1=OrdinalEncoder(categories=[pm])

e1.fit_transform(df[["ord_2"]])
```
![Screenshot 2025-04-18 131501](https://github.com/user-attachments/assets/a540a0c9-e391-4f89-8884-6f510d459a4d)
```
df['bo2']=e1.fit_transform(df[["ord_2"]])

df
```
![Screenshot 2025-04-18 131528](https://github.com/user-attachments/assets/e0a7732c-b472-4e71-bae1-46b693557d92)
```
le=LabelEncoder()

dfc=df.copy()

dfc['ord_2']=le.fit_transform(dfc['ord_2'])

dfc
```
![Screenshot 2025-04-18 131528](https://github.com/user-attachments/assets/2360d4b7-2357-48df-adfd-1d8346fbe2eb)
```
from sklearn.preprocessing import OneHotEncoder

ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()

enc=pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))

df2=pd.concat([df2,enc],axis=1)

df2
```
![Screenshot 2025-04-18 132150](https://github.com/user-attachments/assets/9b539d85-aec7-41d4-977e-b5a3a5356ced)
```
pd.get_dummies(df2,columns=["nom_0"])
```
![Screenshot 2025-04-18 132213](https://github.com/user-attachments/assets/80b78111-8060-4041-91b8-57719f11c81c)
```
pip install --upgrade category_encoders

from category_encoders import BinaryEncoder

df=pd.read_csv("/content/data (2).csv")

df
```
![Screenshot 2025-04-18 134003](https://github.com/user-attachments/assets/9b56ecac-7cf6-4426-b2db-6ef7b659ee62)
```
be=BinaryEncoder()

nd=be.fit_transform(df['Ord_2'])

dfb=pd.concat([df,nd],axis=1)

dfb1=df.copy()

dfb
```
![Screenshot 2025-04-18 134038](https://github.com/user-attachments/assets/4c0a6d59-5ff5-4797-909d-a33765fec996)
```
from category_encoders import TargetEncoder

te=TargetEncoder

cc=df.copy()

te = TargetEncoder()
new=te.fit_transform(X=cc["City"],y=cc["Target"])

cc=pd.concat([cc,new],axis=1)

cc
```
![Screenshot 2025-04-18 134144](https://github.com/user-attachments/assets/f2d97208-aec0-4bcd-b32a-a6ff1ee9188c)
```
import pandas as pd
from scipy import stats
import numpy as np

df=pd.read_csv("/content/Data_to_Transform (1).csv")

df
```
![Screenshot 2025-04-18 134438](https://github.com/user-attachments/assets/bba55358-c3c7-4efb-bc34-ac360aed43d2)
```
df.skew()
```
![Screenshot 2025-04-18 134501](https://github.com/user-attachments/assets/9954163a-7d7b-4cb6-b2c5-4bfd912e5d8e)
```
np.log(df["Highly Positive Skew"])
```
![Screenshot 2025-04-18 134520](https://github.com/user-attachments/assets/8df173cb-ca6c-4031-996c-fcd82d0042c3)
```
np.reciprocal(df["Moderate Positive Skew"])
```
![Screenshot 2025-04-18 134547](https://github.com/user-attachments/assets/a90fe117-4576-4408-bfae-49e9117c5a5b)
```
np.sqrt(df["Highly Positive Skew"])
```
![Screenshot 2025-04-18 134606](https://github.com/user-attachments/assets/506583c0-6796-499c-aedb-5cc1f6a8a91d)
```
np.square(df["Highly Positive Skew"])
```
![Screenshot 2025-04-18 134654](https://github.com/user-attachments/assets/527d7028-279c-49d3-b304-9ff4e10a5fee)
```
df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])

df
```
![Screenshot 2025-04-18 134816](https://github.com/user-attachments/assets/b44528ed-7393-4955-bb5c-201699d0cdd9)
```
df.skew()
```
![Screenshot 2025-04-18 134841](https://github.com/user-attachments/assets/dfa483d1-5fa0-489f-9d62-9984a1cdb159)
```
df["High Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])

df.skew()
```
![Screenshot 2025-04-18 134913](https://github.com/user-attachments/assets/f08b9397-d97f-4d53-9a68-13e32897f9c8)
```
from  sklearn.preprocessing import QuantileTransformer

qt=QuantileTransformer(output_distribution='normal')

df["Moderate Negative Skew_1"] = qt.fit_transform(df[["Moderate Negative Skew"]])

df
```
![Screenshot 2025-04-18 134944](https://github.com/user-attachments/assets/0991bb9a-8f90-42f3-9f8d-ff7123eb4f2c)
```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![Screenshot 2025-04-18 135025](https://github.com/user-attachments/assets/26a33ed2-e9b1-44e8-a646-90dd57ce45a4)
```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
```
![Screenshot 2025-04-18 193753](https://github.com/user-attachments/assets/99911ef2-5196-46ad-b432-29c65056eeb0)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![Screenshot 2025-04-18 193846](https://github.com/user-attachments/assets/d3e3e272-4ad7-4044-a124-c5eb566cb323)
```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df['Highly Negative Skew'],line='45')
plt.show()
```
![Screenshot 2025-04-18 193913](https://github.com/user-attachments/assets/61cd454e-b209-491a-b684-36229cf0a44f)
```
sm.qqplot(df['Highly Negative Skew_1'],line='45')
plt.show()

dt=pd.read_csv("/content/titanic_dataset (2).csv")

from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

dt["Age_1"]=qt.fit_transform(dt[["Age"]])

sm.qqplot(dt["Age"], line='45')
plt.show()
```
![Screenshot 2025-04-18 194126](https://github.com/user-attachments/assets/42f69197-b8ee-483c-97fd-c2ddeaf06fc4)

# RESULT:
Feature Encoding and Transformation process has been successfully performed using
 the data set.

       
