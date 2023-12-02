#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# ### data understanding and exploration

# In[2]:


import numpy as np
import pandas as pd


# In[3]:


dataset=pd.read_csv("BoomBikes.csv")


# In[4]:


dataset


# In[5]:


dataset.shape


# In[6]:


dataset.columns


# In[7]:


dataset.describe()


# In[8]:


dataset.info()


# In[9]:


# assigning string values to seasons
# 1 = spring
dataset.loc[dataset["season"]==1, 'season']='spring'
# 2 = summer
dataset.loc[dataset["season"]==2, 'season']='summer'
# 3 = fall
dataset.loc[dataset["season"]==3, 'season']='fall'
# 4 = winter
dataset.loc[dataset["season"]==4, 'season']='winter'


# In[10]:


dataset["season"].astype("category").value_counts()


# In[11]:


dataset["yr"].astype("category").value_counts()


# In[12]:


def object_map_months(x):
    return x.map({1: "jan", 2: "feb", 3: "mar", 4: "apr", 5: "may", 6: "jun", 7: "jul", 8: "aug", 9: "sep", 10: "oct", 11: "nov", 12: "dec"})


# In[13]:


dataset[["mnth"]]=dataset[["mnth"]].apply(object_map_months)  # 2 parantheisis bcz we have to give value of individual list


# In[14]:


dataset["mnth"].astype("category").value_counts()


# In[15]:


dataset["holiday"].astype("category").value_counts()


# In[16]:


def object_map_weekday(x):
    return x.map({1: "mon", 2: "tue", 3: "wed", 4: "thu", 5: "fri", 6: "sat", 0: "sun"})


# In[17]:


dataset[["weekday"]]=dataset[["weekday"]].apply(object_map_weekday)

temodataset["weekday"].astype("category").value_counts()
# In[18]:


dataset["workingday"].astype("category").value_counts()


# In[19]:


def object_map_weathersit(x):
    return x.map({1: "A", 2: "B", 3: "C"})


# In[20]:


dataset[["weathersit"]]=dataset[["weathersit"]].apply(object_map_weathersit)


# In[21]:


dataset["weathersit"].astype("category").value_counts()


# In[22]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[23]:


sns.distplot(dataset["temp"])


# In[24]:


sns.distplot(dataset["atemp"])


# In[25]:


sns.distplot(dataset["windspeed"])


# In[26]:


sns.distplot(dataset["cnt"])


# In[27]:


import datetime

dataset["dteday"]=dataset["dteday"].apply(lambda x: datetime.datetime.strptime(x, '%d-%m-%Y'))


# In[28]:


dataset_categorical=dataset.select_dtypes(exclude=["float64", "datetime64", "int64"])


# In[29]:


dataset_categorical


# In[30]:


plt.figure(figsize=(20, 20))
plt.subplot(3, 3, 1)
sns.boxplot(x="season", y="cnt", data=dataset)
plt.subplot(3, 3, 2)
sns.boxplot(x="mnth", y="cnt", data=dataset)
plt.subplot(3, 3, 3)
sns.boxplot(x="weekday", y="cnt", data=dataset)
plt.subplot(3, 3, 4)
sns.boxplot(x="weathersit", y="cnt", data=dataset)
plt.subplot(3, 3, 5)
sns.boxplot(x="workingday", y="cnt", data=dataset)
plt.subplot(3, 3, 6)
sns.boxplot(x="yr", y="cnt", data=dataset)
plt.subplot(3, 3, 7)
sns.boxplot(x="holiday", y="cnt", data=dataset)


# In[31]:


intVarlist=["casual", "registered", "cnt"]
for var in intVarlist:
    dataset[var]=dataset[var].astype("float")


# In[32]:


dataset_numeric=dataset.select_dtypes(include=["float64"])


# In[33]:


dataset_numeric


# In[34]:


sns.pairplot(dataset_numeric)


# In[35]:


cor=dataset_numeric.corr()
cor


# In[36]:


mask=np.array(cor)
mask[np.tril_indices_from(mask)]=False
fig, ax=plt.subplots()
fig.set_size_inches(10, 10)
sns.heatmap(cor, mask=mask, vmax=0.8, square=True, annot=True)


# In[37]:


dataset.drop("atemp", axis=1, inplace=True)  # dropping bcz atemp is very closely correlated with temp


# In[38]:


dataset.head()


# #### one hot encoding using pandas

# In[39]:


dataset_categorical=dataset.select_dtypes(include=["object"])


# In[40]:


dataset_categorical.head()


# In[41]:


dataset_dummies=pd.get_dummies(dataset_categorical, drop_first=True)
dataset_dummies.head()


# In[42]:


dataset=dataset.drop(list(dataset_categorical.columns),axis=1)
dataset


# In[43]:


dataset=pd.concat([dataset, dataset_dummies], axis=1)


# In[44]:


dataset.head()


# In[45]:


dataset=dataset.drop(["instant", "dteday"], axis=1, inplace=False)


# In[46]:


dataset.head()


# In[47]:


from sklearn import linear_model
from sklearn.linear_model import LinearRegression


# In[48]:


from sklearn.model_selection import train_test_split
np.random.seed(0)
df_train, df_test = train_test_split(dataset, train_size=0.7, random_state=100)


# In[49]:


df_train


# In[50]:


from sklearn.preprocessing import MinMaxScaler


# In[51]:


scaler=MinMaxScaler()


# In[52]:


var=["temp", "hum", "windspeed", "casual", "registered", "cnt"]
df_train[var]=scaler.fit_transform(df_train[var])


# In[53]:


df_train.describe()


# In[54]:


plt.figure(figsize=(30, 30))
sns.heatmap(df_train.corr(), annot=True, cmap="YlGnBu")


# In[55]:


y_train=df_train.pop("cnt")
x_train=df_train.drop(["registered", "casual"], axis=1).astype(float)  
# removing registred and casual as they have high correlation with cnt, and also [cnt=registred+casual]


# In[56]:


np.array(x_train)


# In[57]:


# Adding Constant Term
import statsmodels.api as sm
x_train_lm=sm.add_constant(x_train)
lr=sm.OLS(y_train, x_train_lm).fit()


# In[58]:


lr.params


# In[59]:


lm=LinearRegression()
lm.fit(x_train, y_train)


# In[60]:


print(lm.coef_)
print(lm.intercept_)


# In[61]:


lr.summary()


# #### to automatically eliminate features which are not significant

# In[62]:


from sklearn.feature_selection import RFE


# In[63]:


lm=LinearRegression()
rfe1=RFE(lm, n_features_to_select=15)
rfe1.fit(x_train, y_train)
print(rfe1.support_)
print(rfe1.ranking_)


# In[64]:


col1=x_train.columns[rfe1.support_]
col1


# In[65]:


x_train_rfe1=x_train[col1]

x_train_lm1=sm.add_constant(x_train_rfe1)
lm1=sm.OLS(y_train, x_train_lm1.astype(float)).fit()
lm1.summary()


# In[66]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[67]:


a=x_train_lm1.drop("const", axis=1)


# In[68]:


# Evaluating Variance Inflation Factor
vif=pd.DataFrame()
vif["features"]=a.columns
vif["VIF"]=[variance_inflation_factor(a.values, i) for i in range(a.shape[1])]
vif["VIF"]=round(vif["VIF"], 2)
vif=vif.sort_values(by="VIF", ascending=False)


# In[69]:


vif


# In[70]:


lm=LinearRegression()
rfe2=RFE(lm, n_features_to_select=7)
rfe2.fit(x_train, y_train)
print(rfe2.support_)
print(rfe2.ranking_)


# In[71]:


col2=x_train.columns[rfe2.support_]

x_train_rfe2=x_train[col2]

x_train_lm2=sm.add_constant(x_train_rfe2)
lm2=sm.OLS(y_train, x_train_lm2).fit()
lm2.summary()


# In[72]:


b=x_train_lm2.drop("const", axis=1)

vif1=pd.DataFrame()
vif1["features"]=b.columns
vif1["VIF"]=[variance_inflation_factor(b.values, i) for i in range(b.shape[1])]
vif1["VIF"]=round(vif["VIF"], 2)
vif1=vif1.sort_values(by="VIF", ascending=False)
vif1


# In[73]:


y_train_cnt=lm2.predict(x_train_lm2)


# In[74]:


fig=plt.figure()
sns.distplot((y_train, y_train_cnt), bins=20)


# In[75]:


df_test[var]=scaler.transform(df_test[var])
df_test


# In[76]:


y_test=df_test.pop("cnt")
x_test=df_test.drop(["casual","registered"], axis=1)


# In[77]:


x_test.head()


# In[78]:


c=x_train_lm2.drop("const", axis=1)


# In[79]:


col2=c.columns


# In[80]:


x_test_rfe2=x_test[col2]


# In[81]:


x_test_lm2=sm.add_constant(x_test_rfe2)


# In[82]:


x_test_lm2.info()


# In[83]:


y_pred=lm2.predict(x_test_lm2)


# In[84]:


plt.figure()
plt.scatter(y_test,y_pred)


# In[85]:


from sklearn.metrics import r2_score
r2_score(y_test, y_pred)


# In[86]:


plt.figure(figsize=(8,5))

sns.heatmap(dataset[col2].corr(),cmap="YlGnBu", annot=True)
plt.show()


# In[ ]:





# In[ ]:




