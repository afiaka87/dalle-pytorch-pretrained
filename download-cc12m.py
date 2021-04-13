#!/usr/bin/env python
# coding: utf-8

# In[6]:


get_ipython().system('apt-get -y install git wget aria2')
get_ipython().run_line_magic('cd', '/workspace/')


# In[9]:


get_ipython().system('wget --no-clobber https://storage.googleapis.com/conceptual_12m/cc12m.tsv')


# In[5]:


get_ipython().system('wget https://storage.googleapis.com/conceptual_12m/cc12m.tsv')


# In[21]:


get_ipython().run_line_magic('cd', '/workspace')

from tqdm import tqdm
with open("/workspace/cc12m.tsv", 'r', encoding='utf-8') as reader, open("/workspace/cc12m_dl.txt", 'w', encoding='utf-8') as writer:
    for i, line in enumerate(reader):
        url = line.split("\t")[0]
        writer.write(url + "\n\tout=" + str(i) + ".jpg\n")


# In[22]:


get_ipython().run_line_magic('mkdir', 'cc12m_images')
get_ipython().run_line_magic('cp', 'cc12m_dl.txt cc12m_images')
get_ipython().run_line_magic('cd', '/workspace/cc12m_images')


# In[ ]:


"cc12m_dl.txt"
get_ipython().system('aria2c -i cc12m_dl.txt -j 16 -q --deferred-input true')

