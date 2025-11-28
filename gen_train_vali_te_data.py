#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[3]:


df = pd.read_csv("./data/data.csv")


# In[4]:


# 分为good和bad两个DataFrame
df_good = df[df['label'] == 'good']
df_bad = df[df['label'] == 'bad']


# In[7]:


# 输出各自的数量
print(f"good数量: {len(df_good)}")
print(f"bad数量: {len(df_bad)}")


# In[8]:


# 随机抽取各60000个
df_good_sample = df_good.sample(n=60000, random_state=42)
df_bad_sample = df_bad.sample(n=60000, random_state=42)


# In[10]:


# good分为训练、验证、测试
df_good_train = df_good_sample.sample(n=40000, random_state=42)
df_good_cv = df_good_sample.drop(df_good_train.index).sample(n=10000, random_state=42)
df_good_test = df_good_sample.drop(df_good_train.index).drop(df_good_cv.index)

# 打印数量
print(f"good训练集数量: {len(df_good_train)}")
print(f"good验证集数量: {len(df_good_cv)}")
print(f"good测试集数量: {len(df_good_test)}")


# In[11]:


# bad分为训练、验证、测试
df_bad_train = df_bad_sample.sample(n=40000, random_state=42)
df_bad_cv = df_bad_sample.drop(df_bad_train.index).sample(n=10000, random_state=42)
df_bad_test = df_bad_sample.drop(df_bad_train.index).drop(df_bad_cv.index)

# 打印数量
print(f"bad训练集数量: {len(df_bad_train)}")
print(f"bad验证集数量: {len(df_bad_cv)}")
print(f"bad测试集数量: {len(df_bad_test)}")


# In[12]:


df_train = pd.concat([df_good_train, df_bad_train], axis=0)
df_cv = pd.concat([df_good_cv, df_bad_cv], axis=0)
df_test = pd.concat([df_good_test, df_bad_test], axis=0)


# In[13]:


df_train


# In[14]:


# 打乱
df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)
df_cv = df_cv.sample(frac=1, random_state=42).reset_index(drop=True)
df_test = df_test.sample(frac=1, random_state=42).reset_index(drop=True)


# In[15]:


# 只保留URL和label两列
df_train = df_train[['URL', 'label']]
df_cv = df_cv[['URL', 'label']]
df_test = df_test[['URL', 'label']]


# In[17]:


# 保存
df_train.to_csv("./data/merged_train.csv", index=False)
df_cv.to_csv("./data/merged_cv.csv", index=False)
df_test.to_csv("./data/merged_test.csv", index=False)


# In[ ]:




