
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sys

# In[4]:

df = pd.read_csv("year_data.csv")
df


# In[12]:

# !!!change zip code manually hear!!! 
zip_code = sys.argv[1]
year = df[zip_code]


# In[13]:

#plt.plot(df["zip_code"], year, 'ro')


# In[8]:

x = np.arange(len(df))
#print(np.arange(len(df)))
#xi = df["zip_code"]


# In[16]:

df[zip_code]


# In[9]:

slope, intercept, r_value, p_value, std_err = stats.linregress(x, year)
line1 = intercept + slope*x
plt.plot(line1,'r-')


# In[17]:

x_labels = df[zip_code]

slope, intercept, r_value, p_value, std_err = stats.linregress(x, year)
line1 = intercept + slope*x

plt.plot(x,  year, 'ro')
plt.xticks (x, x_labels)
plt.subplots_adjust(bottom=0.15)
plt.plot(line1,'r-')
plt.show()
data = np.array(r_value)
data = np.append(data, r_value**2)
data = np.append(data, p_value)
print(data)


# In[20]:

user_year = int(input("Year to predict: "))
print("ZIP CODE : ", zip_code)
print("Predicted income : ", (user_year*slope) + intercept)


# In[214]:

#print(r_value)


# In[215]:

#print(r_value**2)
#sqr_r = r_value**2


# In[216]:

#print(p_value)


# In[19]:

#print(2100*slope + intercept)


# In[213]:

#data_list = []
#data_list.append(r_value)
#data_list.append(r_value**2)
#data_list.append(p_value)
#print(data_list)
#data = np.array(r_value)
#data = np.append(data, r_value**2)
#data = np.append(data, p_value)
#print(data)


# In[207]:

#print(df.ix[0, "0":"21921"])
#df.loc["zip_code":]


# In[205]:

#new_df = pd.DataFrame()
#columns = ["r_value", "r_value**2", "p_value"]
#index = df.loc["zip_code":]
#new_df = pd.DataFrame(data, index=index, columns=columns)

