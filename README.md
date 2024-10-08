# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/bmi.csv")
df.head()
```

![image](https://github.com/user-attachments/assets/7506dd32-2b84-4112-b3be-abaea0fdbde4)
```
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)
```

![image](https://github.com/user-attachments/assets/e480b4c5-c9ce-44cb-91fc-21210672f002)

```
max_vals=np.max(np.abs(df[['Height','Weight']]))
max_vals
```

![image](https://github.com/user-attachments/assets/221ca5c9-1e76-473d-b6bf-84b44183e4c8)

```
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```

![image](https://github.com/user-attachments/assets/489d349b-9331-4921-b9cd-79aed358e37e)

```
from sklearn.preprocessing import Normalizer
scaler=Normalizer()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```

![image](https://github.com/user-attachments/assets/8c4a4f05-857c-48ae-a307-485c5a200040)

```
from sklearn.preprocessing import MaxAbsScaler
sc=MaxAbsScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df
```

![image](https://github.com/user-attachments/assets/6f78a172-0ffe-4e5c-9dc7-9cd79ac4f31d)

```
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head()
```

![image](https://github.com/user-attachments/assets/1edefc58-a821-4a9e-af17-7e521c41f97e)

##FEATURE SELECTION
```
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
```
```
df=pd.read_csv("/content/income(1) (1) (1).csv",na_values=[" ?"])
df
```

![image](https://github.com/user-attachments/assets/feb77bfb-7830-4d78-8cc8-014572855892)

```
df.isnull().sum()
```

![image](https://github.com/user-attachments/assets/06f5b281-6ff7-4301-a8ca-cd22beb16709)

```
miss=df[df.isnull().any(axis=1)]
miss
```

![image](https://github.com/user-attachments/assets/86d3fa5e-2a4b-4936-9212-8910b8b2897b)

```
df2=df.dropna(axis=0)
df2
```

![image](https://github.com/user-attachments/assets/99efded2-ec15-4983-b9c7-139998945075)

```
sal=df['SalStat']
df2['SalStat']=df2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(df2['SalStat'])
```

![image](https://github.com/user-attachments/assets/04766e39-7d2a-4fe6-a65b-d9e74f53ed31)

```
sal2=df2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```

![image](https://github.com/user-attachments/assets/1b6246ba-6c81-43f3-be63-44445c61bf76)

```
df2
```

![image](https://github.com/user-attachments/assets/c79e9135-f313-4359-ab8d-64e04a23cf70)

```
new_data=pd.get_dummies(df2,drop_first=True)
new_data
```

![image](https://github.com/user-attachments/assets/2e4e8db7-9360-447b-b5fc-8fe0de1a1600)

```
columns_list=list(new_data.columns)
print(columns_list)
```

![image](https://github.com/user-attachments/assets/945603b5-9ae6-46b4-9ba0-ba5b20a13a31)

```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```

![image](https://github.com/user-attachments/assets/d08a59d1-17ee-44b7-b6cd-b79a7919f5cc)

```
y=new_data['SalStat'].values
print(y)
```

![image](https://github.com/user-attachments/assets/b30cdfb1-adee-448f-9621-f45a2b9f9024)

```
x=new_data[features].values
print(x)
```

![image](https://github.com/user-attachments/assets/e6ddeb1b-5d50-4e61-942b-824e924854ad)

```
train_x, test_x, train_y, test_y=train_test_split(x,y,test_size=0.3, random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors=5)
KNN_classifier.fit(train_x, train_y)
```

![image](https://github.com/user-attachments/assets/8575f489-6cae-43bc-9c20-6ae1b1abdd8b)

```
prediction=KNN_classifier.predict(test_x)
confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```

![image](https://github.com/user-attachments/assets/1a8ac4c9-9f95-4baa-ab15-21165b9c45a2)

```
accuracy_score=accuracy_score(test_y, prediction)
print(accuracy_score)
```

![image](https://github.com/user-attachments/assets/722f6080-e786-4d59-9b6a-c904ac741fa5)

```
print('Misclassified samples: %d' % (test_y !=prediction).sum())
```

![image](https://github.com/user-attachments/assets/32a8dcf5-d7c7-4a3c-806f-dbeb168d3f90)

```
df.shape
```

![image](https://github.com/user-attachments/assets/ed2d0435-a535-4024-af17-13d5fdbb4584)

```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset("tips")
tips.head()
```

![image](https://github.com/user-attachments/assets/93b70137-7e70-4339-9f30-39a1e43896d1)

```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```

![image](https://github.com/user-attachments/assets/747cdbba-347e-4f6a-ad40-4bbcdcf4c5b4)

```
schi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```

![image](https://github.com/user-attachments/assets/71211cf0-f69d-4bab-90fc-cf7ccabc9abb)

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1': [1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3': [0,1,1,0,1],
    'Target'  : [0,1,1,0,1]
}
df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]
selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![image](https://github.com/user-attachments/assets/85a863fb-1860-4dc4-82ef-a8187fc56e2f)


# RESULT:
       Thus, Feature Scaling and Feature Selection process have been done successfully.
