
# coding: utf-8

# In[4]:

import os
import pandas as pd
import numpy as np
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt

data = pd.read_csv("data.csv")


# In[7]:

data = data.drop("ave",1)


# In[8]:

data


# In[9]:

from sklearn.tree import DecisionTreeClassifier, export_graphviz


# In[10]:

df2 = data.copy()


# In[11]:

# df2.apply(lambda x: pd.to_numeric(x, downcast='float'))
df2 = df2.astype(float)
df2[:5]


# In[12]:

df2.columns


# In[13]:

from sklearn import preprocessing
le = preprocessing.LabelEncoder()


# In[16]:

#put money in 1998 currency terms
classifier = le.fit_transform(df2['1998'])


# In[17]:

le.inverse_transform([0,1,2,3,4,5,6,7,8,9,10])


# In[18]:

dt = DecisionTreeClassifier()


# In[19]:

dt.fit(df2.values, classifier)


# In[20]:

from IPython.display import Image


# In[21]:

dt = DecisionTreeClassifier()


# In[22]:

dt.fit(df2.values, classifier)
with open("dt.dot", 'w') as f:
    export_graphviz(dt, out_file=f, feature_names=df2.columns)


# In[23]:

os.system('dot -Tpng dt.dot -o dt.png')


# In[25]:

#Image(filename='dt.png')


# In[323]:

#graph is too large! It overfitted to the data and was unable to create something useful


# In[324]:

dt = DecisionTreeClassifier(max_depth = 100)


# In[325]:

dt.fit(df2.values, classifier)
with open("dt.dot", 'w') as f:
    export_graphviz(dt, out_file=f, feature_names=df2.columns)


# In[26]:

# part 2

data2 = data.copy()


# In[27]:

data2[:5]


# In[28]:

data2["mns_sqrt_root"] =  (data["1998"])**2 + (data["2001"])**2 + (data["2004"])**2 + (data["2005"])**2+ (data["2006"])**2 + (data["2007"])**2 + (data["2010"])**2  + (data["2011"])**2 + (data["2012"])**2 + (data["2013"])**2 + (data["2014"])**2     


# In[29]:

import math
#data2["mns_sqrt_root"] = data2["mns_sqrt_root"].apply(lambda s: math.sqrt(s))


# In[30]:

data2.head(5)


# In[31]:

data2 = data2.astype(float)


# In[32]:

data2[400:]


# In[33]:

classifier = le.fit_transform(data2['mns_sqrt_root'])


# In[34]:

le.inverse_transform(range(0,402))


# In[35]:

dt = DecisionTreeClassifier(max_depth = 20)


# In[36]:

dt.fit(data2.values, classifier)
with open("dt.dot", 'w') as f:
    export_graphviz(dt, out_file=f, feature_names=data2.columns)


# In[37]:

# Part 3


# In[39]:

data3 = pd.read_csv("flipped_data.csv")


# In[40]:

data3.head()


# In[41]:

data3 = data3.drop('zip_code',1)


# In[42]:

data3 = data3.astype(float)


# In[43]:

classifier = le.fit_transform(data3['Totals'])


# In[44]:

le.inverse_transform(range(0,11))


# In[45]:

dt = DecisionTreeClassifier()


# In[46]:

dt.fit(data3.values, classifier)
with open("dt.dot", 'w') as f:
    export_graphviz(dt, out_file=f, feature_names=data3.columns)


# In[52]:

#df = pd.read_csv("Better_14zp21md_cleaned.csv")


# In[53]:

#df = df.drop("Unnamed: 0",1)


# In[56]:

#df[:2]


# In[58]:

#This was not a great idea ^^


# In[89]:

#df = pd.read_csv("flipped_data.csv")


# In[87]:

#df


# In[88]:

#df = df.drop("zip_code",1)
#df = df.drop("Totals",1)


# In[66]:

#df = df.drop("Totals",1)


# In[86]:

#df


# In[85]:

#df2 = df.copy()


# In[84]:

#df2.pct_change(periods=3)


# In[83]:

#df2


# In[82]:

#data = np.matrix( np.asarray( df2 ) )


# In[72]:

from sklearn.linear_model import LinearRegression


# In[81]:

#lr = LinearRegression()


# In[80]:

# X, y = data[:, 1:], data[:, 0]


# In[79]:

#lr.fit(X, y)
#LinearRegression(copy_X=True, fit_intercept=True, normalize=False)


# In[76]:

#from array import array


# In[78]:

#lr.coef_array([  4.01182386e-01,   3.51587361e-04])


# In[99]:

df = pd.read_csv("flipped_data.csv")
df = df.drop("zip_code",1)
#df = df.drop("Totals",1)


# In[100]:

#df


# In[101]:

import numpy as np
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.distributions.mixture_rvs import mixture_rvs


# In[102]:

np.random.seed(12345)


# In[103]:

mixture_rvs([.25,.75], size=11, dist=[stats.norm, stats.norm],
               kwargs = (dict(loc=-1,scale=.5),dict(loc=1,scale=.5)))


# In[104]:

obs_dist1 = mixture_rvs([.25,.75], size=10000, dist=[stats.norm, stats.norm],
               kwargs = (dict(loc=-1,scale=.5),dict(loc=1,scale=.5)))
df.astype(float)
#data = np.asarray( df2["Totals"] ) 


# In[105]:

data #= data.astype(float)


# In[106]:

obs_dist1


# In[107]:

kde = sm.nonparametric.KDEUnivariate(data)
#kde.fit()


# In[108]:

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
ax.hist(obs_dist1, bins=50, normed=True, color='red')
#ax.plot(kde.support, kde.density, lw=2, color='black');


# In[109]:

fig
#print(fig)
fig.savefig("KDE_estimate_ofdata.png")
# In[ ]:





# In[3]:

#from pandas.io.data import DataReader


# In[2]:

#symbols = ['MSFT', 'GOOG', 'AAPL']
#data = dict((sym, DataReader(sym, "yahoo"))
#            for sym in symbols)


# In[1]:

#panel = Panel(data).swapaxes('items', 'minor')


# In[ ]:



