#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Converting log file to csv
import re
def extract(filename):
    with open(filename) as f:
        log = f.read()
        regexp2 = r"(?P<ip>.*?) (?P<remote_log_name>.*?) (?P<userid>.*?) \[(?P<date>.*?) (?P<timezone>.*?)\] \"(?P<request_method>.*?) (?P<path>.*?) (?P<request_version>.*?)\" (?P<status>.*?) (?P<length>.*?) \"(?P<referrer>.*?)\" \"(?P<user_agent>.*?)\""
        ips_list = re.findall(regexp2, log)
        return ips_list
logs = extract('access.log')
import numpy as np
log_arr = np.array(logs)
ip=log_arr[:,0]
import pandas as pd
dataset = pd.DataFrame({'IP': log_arr[:, 0], 'A': log_arr[:, 1],'B':log_arr[:, 2],'Date&Time':log_arr[:, 3],'TZ':log_arr[:, 4],'C':log_arr[:, 5],'Site':log_arr[:, 6],'Protocol':log_arr[:, 7],'Status':log_arr[:, 8],'Length':log_arr[:, 9]})
print(dataset.head())
dataset.to_csv('weblog.csv')


# In[2]:


import pandas as pd
dataset=pd.read_csv('weblog.csv')
dataset.head()


# In[3]:


import numpy as np
dataset=dataset[['IP','Status','Length']]
dataset.head()
#data=dataset.values


# In[4]:


ip=dataset.IP.unique().tolist()
dataset['Code']=np.zeros(len(dataset))
for i in range(len(dataset)):
    dataset.Code[i]=ip.index(dataset.IP[i])
  
dataset.head()


# In[5]:


data=dataset[["Code","Status"]]
df=data.copy()
df['Code'] = df['Code'].astype(int)
print(df.dtypes)
log=df.values
log


# In[ ]:





# In[ ]:





# In[ ]:





# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_scaled = scaler.fit_transform(log)
pd.DataFrame(data_scaled).describe()
SSE = []
for cluster in range(1,20):
    kmeans = KMeans(n_jobs = -1, n_clusters = cluster, init='k-means++')
    kmeans.fit(data_scaled)
    SSE.append(kmeans.inertia_)
frame = pd.DataFrame({'Cluster':range(1,20), 'SSE':SSE})
plt.figure(figsize=(12,6))
plt.plot(frame['Cluster'], frame['SSE'], marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
kmeans = KMeans(n_jobs = -1, n_clusters = 5, init='k-means++')
kmeans.fit(data_scaled)
pred = kmeans.predict(data_scaled)
frame = pd.DataFrame(data_scaled)
frame['cluster'] = pred
frame['cluster'].value_counts()


# In[7]:


plt.figure(figsize=(12,6))
plt.scatter(dataset['IP'],dataset["Status"],marker='*')
plt.xlabel('IP')
plt.ylabel('Status code')
plt.show()


# In[ ]:





# In[12]:


blocked=[]
for i in range(len(dataset)):
    if dataset['Code'][i]==401 or dataset['Code'][i]==403:
        blocked.append(dataset['IP'][i])
print(blocked)
with open('blocked.html', 'w') as filehandle:
    for listitem in blocked:
        filehandle.write('=>%s\t\t' % listitem)


# In[ ]:





# In[ ]:




