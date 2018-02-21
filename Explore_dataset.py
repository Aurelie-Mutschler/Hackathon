
# coding: utf-8

# In[2]:


import pandas as pd
from plotnine import *
from termcolor import colored, cprint


# In[3]:


data = pd.read_csv('dataset_with_labels.csv').drop('Unnamed: 0',axis=1).sample(n=5000).reset_index(drop=True)
print(data.columns)


# In[4]:


data.loc[:,'diagnosis'] = data['diagnosis'].astype('str')
for c,col in data.iteritems():
    
    Sbold = lambda x : colored(x, attrs=['bold'])
    Scyan_h = lambda x : colored(x, 'grey', 'on_cyan', attrs=['bold'])
    Sred = lambda x : colored(x, 'red', attrs=['bold'])
    Sred_h = lambda x : colored(x, 'grey', 'on_red', attrs=['bold'])
    Sgreen = lambda x : colored(x, 'green', attrs=['bold'])
    
    print('-----------------------------')
    
    print('Feature name : ' + Scyan_h(c))
    
    # Variable type
    var_type = type(col[0]).__name__
    print('Type : ' + Sbold(var_type))
    
    # Rate of missing values
    rate_na = 100*col.isnull().sum()/data.shape[0]
    if rate_na==100:
        print('Rate of missing values : ' + Sred_h(str(int(rate_na))+ '%'))
        print(Sred_h('This column is empty !'))
        # If the column is empty, don't do further tests
        continue
    elif rate_na==0:
        print('Rate of missing values : ' + Sgreen(str(int(rate_na))+ '%'))
    elif (rate_na>=50) & (rate_na<100):
        print('Rate of missing values : ' + Sred(str(int(rate_na))+ '%'))
    else:
        print('Rate of missing values : ' + Sbold(str(int(rate_na))+ '%'))
    
    # Number of unique values
    val_counts = col.value_counts()
    if val_counts.shape[0]==1:
        print('Number of unique values : ' + Sred_h(str(val_counts.shape[0])))
        print(Sred_h('This column is useless !'))
        # If the column is useless, don't do further tests
        continue
    else:
        print('Number of unique values : ' + Sbold(str(val_counts.shape[0])))
    
    # Basic statistics and plots for numerical values
    if var_type in ['int64','float64']:
        # Stats
        print('Median : ' + Sbold(int(col.median())))
        print('Mean : ' + Sbold(int(col.mean())))
        print('[Min, Max] : ' + Sbold('[' + str(int(col.min())) + ', ' + str(int(col.max())) + ']'))
        
        # Histogram
        text_size = 12
        if 'delay' in c: text_size = 10
        p = ggplot(aes(x=c), data=data.dropna(axis=0,subset=[c])) + geom_histogram(bins=50)         + theme_xkcd(length=30)
        print(p)
        
    # Basic statistics and plots for non-numerical values
    if var_type=='str':
        # Stats
        print('Mode : ' + Sbold(col.value_counts().index[0]))
        
        # Bar chart
        if val_counts.shape[0]<50:
            p = ggplot(aes(x=c, fill=c), data=data.dropna(axis=0,subset=[c])) + geom_bar()             + theme_xkcd(scale=0.5)
            print(p)
        else:
            print('10 Most frequent values : ')
            for i,val in val_counts[0:10].iteritems():
                print(Sbold(str(i) + ' (' + str(val) + ')'))
        
    


# In[5]:


print(data['diagnosis'].value_counts())

