
# coding: utf-8

# In[308]:

import os
import pandas as pd
import numpy as np
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt

data = pd.read_csv("/Users/evilfantasies/workspace/AI_Linear_Regression/test/data.csv")

data


# In[309]:

from sklearn.tree import DecisionTreeClassifier, export_graphviz


# In[310]:

df2 = data.copy()


# In[ ]:




# In[ ]:




# In[311]:

# df2.apply(lambda x: pd.to_numeric(x, downcast='float'))
df2 = df2.astype(float)
df2[:5]


# In[312]:

df2.columns


# In[313]:

from sklearn import preprocessing
le = preprocessing.LabelEncoder()


# In[314]:

classifier = le.fit_transform(df2['1998'])


# In[315]:

le.inverse_transform([0,1,2,3,4,5,6,7,8,9,10])


# In[ ]:




# In[316]:

dt = DecisionTreeClassifier()


# In[317]:

dt.fit(df2.values, classifier)


# In[ ]:




# In[ ]:




# In[318]:

from IPython.display import Image


# In[319]:

dt = DecisionTreeClassifier()


# In[320]:

dt.fit(df2.values, classifier)
with open("dt.dot", 'w') as f:
    export_graphviz(dt, out_file=f, feature_names=df2.columns)


# In[321]:

os.system('dot -Tpng dt.dot -o dt.png')


# In[322]:

#Image(filename='dt.png')


# In[323]:

#graph is too large! It overfitted to the data and was unable to create something useful


# In[324]:

dt = DecisionTreeClassifier(max_depth = 100)


# In[325]:

dt.fit(df2.values, classifier)
with open("dt.dot", 'w') as f:
    export_graphviz(dt, out_file=f, feature_names=df2.columns)


# In[326]:

# part 2

data2 = pd.read_csv("/Users/evilfantasies/workspace/AI_Linear_Regression/test/data.csv")


# In[327]:

data2[:5]


# In[328]:

data2["mns_sqrt_root"] =  (data["1998"])**2 + (data["2001"])**2 + (data["2004"])**2 + (data["2005"])**2+ (data["2006"])**2 + (data["2007"])**2 + (data["2010"])**2  + (data["2011"])**2 + (data["2012"])**2 + (data["2013"])**2 + (data["2014"])**2     


# In[329]:

import math
data2["mns_sqrt_root"] = data2["mns_sqrt_root"].apply(lambda s: math.sqrt(s))


# In[330]:

data2.head(5)


# In[331]:

data2 = data2.astype(float)


# In[332]:

data2[400:]


# In[333]:

classifier = le.fit_transform(data2['mns_sqrt_root'])


# In[334]:

le.inverse_transform(range(0,402))


# In[335]:

dt = DecisionTreeClassifier(max_depth = 20)


# In[336]:

dt.fit(data2.values, classifier)
with open("dt.dot", 'w') as f:
    export_graphviz(dt, out_file=f, feature_names=data2.columns)


# In[337]:

# Part 3


# In[343]:

data3 = pd.read_csv("/Users/evilfantasies/Desktop/flipped_data.csv")


# In[344]:

data3.head()


# In[346]:

data3 = data3.drop('zip_code',1)


# In[347]:

data3 = data3.astype(float)


# In[350]:

classifier = le.fit_transform(data3['Totals'])


# In[352]:

le.inverse_transform(range(0,11))


# In[353]:

dt = DecisionTreeClassifier()


# In[354]:

dt.fit(data3.values, classifier)
with open("dt.dot", 'w') as f:
    export_graphviz(dt, out_file=f, feature_names=data3.columns)


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



