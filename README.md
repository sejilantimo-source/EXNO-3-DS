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
df=pd.read_csv("Encoding Data.csv")
df
```
<img width="484" height="447" alt="image" src="https://github.com/user-attachments/assets/28f41142-b9a7-4022-a01f-6c3f7955fb1d" />

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
<img width="266" height="237" alt="image" src="https://github.com/user-attachments/assets/e2b54515-ab44-4fae-8f54-acf4abe4c973" />

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
<img width="475" height="441" alt="image" src="https://github.com/user-attachments/assets/792f121a-568d-4483-b1ad-e1c9347dfd9f" />

```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```

<img width="504" height="435" alt="image" src="https://github.com/user-attachments/assets/7f7dc2a4-50c3-4a8b-a136-215933f4d3f1" />

```
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

ohe = OneHotEncoder(sparse_output=False)   # use sparse_output instead of sparse
df2 = df.copy()
enc = pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]),
                   columns=ohe.get_feature_names_out(["nom_0"]))  # add column names
df2 = pd.concat([df2, enc], axis=1)
df2
```

<img width="799" height="451" alt="image" src="https://github.com/user-attachments/assets/ad862be7-fc21-4ba3-843d-2886ae6c9a47" />

```
pd.get_dummies(df2,columns=["nom_0"])
```

<img width="1101" height="438" alt="image" src="https://github.com/user-attachments/assets/847acc3a-e6a4-45bf-90df-ddcdc622040a" />

```
from category_encoders import BinaryEncoder
import pandas as pd

df = pd.read_csv("data.csv")
be = BinaryEncoder()
nd = be.fit_transform(df[['Ord_2']])   # Pass as DataFrame, not Series
dfb = pd.concat([df, nd], axis=1)
dfb
```

<img width="923" height="443" alt="image" src="https://github.com/user-attachments/assets/d9985e2e-c529-44c7-b046-0260d60b5830" />

```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC

```

<img width="737" height="452" alt="image" src="https://github.com/user-attachments/assets/fcd62f08-852b-4cc8-a2d2-5bc5dd62d86f" />

```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df

```

<img width="759" height="546" alt="image" src="https://github.com/user-attachments/assets/60966c9a-ad47-4336-8357-2d57a15c24bb" />

```
df.skew()

```

<img width="385" height="244" alt="image" src="https://github.com/user-attachments/assets/4ac8c73a-890a-424f-a01b-4e769d44da42" />

```
np.log(df["Highly Positive Skew"])

```

<img width="349" height="562" alt="image" src="https://github.com/user-attachments/assets/34f1f212-a823-4897-a2b9-0d2a76ee4476" />

```
np.reciprocal(df["Moderate Positive Skew"])
```

<img width="396" height="559" alt="image" src="https://github.com/user-attachments/assets/6ca9641a-bad7-4d10-908d-66b608179de9" />

```
np.sqrt(df["Highly Positive Skew"])

```

<img width="325" height="548" alt="image" src="https://github.com/user-attachments/assets/0350a604-d6d1-4182-91dc-2f4c29b75078" />

```
np.square(df["Highly Positive Skew"])

```

<img width="324" height="556" alt="image" src="https://github.com/user-attachments/assets/72c49079-417c-4610-b6ae-3ff12ef99863" />

```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df

```

<img width="732" height="563" alt="image" src="https://github.com/user-attachments/assets/4230d9e7-c8fb-4d12-ac8a-f69c587fbf84" />

```
df.skew()

```

<img width="419" height="275" alt="image" src="https://github.com/user-attachments/assets/edf484f0-a160-4ea6-8da3-c3086a6bcb9a" />

```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()

```

<img width="476" height="341" alt="image" src="https://github.com/user-attachments/assets/e414a1c7-2aee-451c-842f-ac2da1c2d99b" />

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df

```

<img width="1695" height="537" alt="image" src="https://github.com/user-attachments/assets/c99b33ee-436b-4ed1-94f3-b8925b4de76a" />

```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()

```

<img width="745" height="561" alt="image" src="https://github.com/user-attachments/assets/78b52b5f-8cde-4fc0-9cc1-447379ff56b0" />

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()

```

<img width="731" height="560" alt="image" src="https://github.com/user-attachments/assets/9e306466-e678-44ce-94c1-6cb8c8b59138" />

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()


```

<img width="707" height="540" alt="image" src="https://github.com/user-attachments/assets/1f7bd0cd-8595-4beb-8e70-db27f58671e2" />

```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()

```

<img width="706" height="547" alt="image" src="https://github.com/user-attachments/assets/31ae6b82-b5ae-4ec1-a249-1fa5e29645ff" />

```
dt=pd.read_csv("titanic_dataset.csv")
dt
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45')
plt.show()

```

<img width="700" height="549" alt="image" src="https://github.com/user-attachments/assets/55baa019-4ae8-465f-b1cf-7ef33e5cff26" />

```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()

```

<img width="724" height="554" alt="image" src="https://github.com/user-attachments/assets/2482cbc2-933c-47b3-9fad-7428a3ea9a91" />

# RESULT:
Thus the given data, Feature Encoding, Transformation process and save the data to a file
was performed successfully.       
       

