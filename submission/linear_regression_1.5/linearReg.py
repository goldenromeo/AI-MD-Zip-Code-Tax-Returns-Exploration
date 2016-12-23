
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# In[2]:

df = pd.read_csv("year_data.csv")
#df


# In[3]:

zip_code = input("Enter any zip code in Maryland : ")
#zip_code = "21921"
year = df[zip_code]


# In[4]:

#plt.plot(df["zip_code"], year, 'ro')


# In[5]:

x = np.arange(len(df))
#print(np.arange(len(df)))
#xi = df["zip_code"]


# In[6]:

slope, intercept, r_value, p_value, std_err = stats.linregress(x, year)
line1 = intercept + slope*x
plt.plot(line1,'r-')


# In[16]:

x_labels = df[zip_code]

slope, intercept, r_value, p_value, std_err = stats.linregress(x, year)
line1 = intercept + slope*x

plt.plot(x,  year, 'ro')
plt.xticks (x, x_labels)
plt.subplots_adjust(bottom=0.15)
plt.plot(line1,'r-')
plt.title("zip code : " + zip_code)
plt.show ()
data = np.array(r_value)
data = np.append(data, r_value**2)
data = np.append(data, p_value)
print(data)


# In[17]:

user_year = int(input("enter a year to predict: "))
print("ZIP CODE : ", zip_code)
print("Predicted total income of the zip code : ", (user_year*slope) + intercept)


# In[ ]:

#print(r_value)


# In[ ]:

#print(r_value**2)
#sqr_r = r_value**2


# In[216]:

#print(p_value)


# In[264]:

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

