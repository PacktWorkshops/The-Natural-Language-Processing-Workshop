# coding: utf-8

# In[21]:


import pandas as pd
data = pd.read_excel('../data/Online Retail.xlsx')
assert data.shape == tuple([541909, 8])
print(data.shape)


# In[23]:


data_sample_random = data.sample(frac=0.1,random_state=42) # selecting 10% of the data randomly
assert data_sample_random.shape == tuple([54191, 8])
print(data_sample_random.shape)


# In[43]:


from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(data, data['Country'],test_size=0.2, random_state=42, stratify = data['Country'])
assert (round((y_train.value_counts()/y_train.shape[0])[:5],3) == round((y_valid.value_counts()/y_valid.shape[0])[:5],3)).all()


# In[44]:


print(y_train.value_counts()/y_train.shape[0])


# In[45]:


print(y_valid.value_counts()/y_valid.shape[0])


# In[46]:


data_ugf = data[data['Country'].isin(['United Kingdom', 'Germany', 'France'])]
data_ugf_q2 = data_ugf[data_ugf['Quantity']>=2]
data_ugf_q2_sample = data_ugf_q2.sample(frac = .02, random_state=42)
assert data_ugf_q2.shape == tuple([356940, 8])
assert data_ugf_q2_sample.shape == tuple([7139, 8])


# In[47]:


print(data_ugf_q2.shape)


# In[48]:


print(data_ugf_q2_sample.shape)
