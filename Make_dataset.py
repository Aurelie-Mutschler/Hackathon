
# coding: utf-8

# In[1]:


# Import libraries
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import psycopg2

# below imports are used to print out pretty pandas dataframes
from IPython.display import display, HTML

get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')


# In[2]:


# information used to create a database connection
sqluser = 'postgres'
dbname = 'mimic'
schema_name = 'mimiciii'

# Connect to postgres with a copy of the MIMIC-III database
con = psycopg2.connect(dbname=dbname, user=sqluser)

# the below statement is prepended to queries to ensure they select from the right schema
query_schema = 'set search_path to ' + schema_name + ';'


# In[3]:


def apply_inclusion_criteria(df):
    print('Initial size of table : ' + str(df.shape[0]))
    df = df.drop_duplicates(['icustay_id','creatinine_time'])
    print('After dropping duplicates : ' + str(df.shape[0]))
    criteria_list = [i for i in df.columns.values if 'inclusion' in i]
    for c in criteria_list:
        df = df.loc[df[c]==1,:].drop(c,axis=1)    
    print('After applying inclusion criteria : ' + str(df.shape[0]))
    return df


# # Load file with features to be found in chartevents table

# In[4]:


# Read file
features_info = pd.read_csv('features_info.csv')
# Drop lines with no item_id
item_col = [c for c in features_info.columns.values if "item" in c]
features_info = features_info.dropna(axis=0, how='all', subset=item_col).reset_index(drop=True)
features_info.head()


# # Retrieve interesting features for patients that match inclusion criteria

# In[5]:


# From features_info, retrieve the list of item_id to use with chartevents
item_col = [c for c in features_info.columns.values if "item_id" in c]
item_list = features_info.loc[0,item_col].dropna().astype('int').values
item_str = "(" + str(item_list[0])
for it in item_list[1:]:
    item_str = item_str + "," + str(it)
item_str = item_str + ")"

# List of ICD-9 codes to be excluded
icd_list = ['5856','V420','99681'] # ESRD: 5856 / kidney transplant: V420,99681
icd_str = "'{" + str(icd_list[0])
for icd in icd_list[1:]:
    icd_str = icd_str + "," + str(icd)
icd_str = icd_str + "}'"
print(icd_str)

query = query_schema + """
with cr1 as
(
select
    icu.subject_id, icu.icustay_id, icu.intime, icu.outtime, EXTRACT(EPOCH FROM icu.outtime - icu.intime)/60.0/60.0 as length_of_stay,
    ce.valuenum as creatinine, ce.storetime as creatinine_time,
    EXTRACT('epoch' from icu.intime - pat.dob) / 60.0 / 60.0 / 24.0 / 365.242 AS age,
    (rrt.icustay_id is null) as rrt,
    diag.icd9_code as diagnosis 
  from icustays icu
  inner join chartevents ce
    on icu.subject_id = ce.subject_id
    and ce.itemid = 220615
    and ce.valuenum is not null
    and ce.storetime between icu.intime and icu.outtime
  inner join patients pat
    ON icu.subject_id = pat.subject_id
  left outer join rrt 
    on icu.icustay_id = rrt.icustay_id
  inner join diagnoses_icd diag
    on icu.subject_id = diag.subject_id
    and icu.hadm_id = diag.hadm_id
),
cr as
(
select
cr1.subject_id, cr1.icustay_id, cr1.intime, cr1.outtime,
cr1.creatinine, cr1.creatinine_time,
cr1.age,
cr1.length_of_stay,
cr1.rrt,
array_agg(cr1.diagnosis) as diagnoses
from cr1
group by cr1.subject_id, cr1.icustay_id, cr1.intime, cr1.outtime, cr1.creatinine, cr1.creatinine_time, cr1.age,
cr1.length_of_stay,cr1.rrt
),
cr_inc as
(
select
cr.subject_id, cr.icustay_id, cr.intime, cr.outtime,
    cr.creatinine, cr.creatinine_time,
    cr.age, CASE
                WHEN cr.age >= 15 then 1
            ELSE 0 END
            as inclusion_age,
  cr.length_of_stay, CASE
                        WHEN cr.length_of_stay >= 48 then 1
                     ELSE 0 END
                     as inclusion_length_of_stay,
  cr.rrt, CASE
            WHEN cr.rrt = False then 1
          ELSE 0 END
          as inclusion_rrt,
  cr.diagnoses, CASE
            WHEN cr.diagnoses && """ + icd_str + """ then 0
          ELSE 1 END
          as inclusion_diagnoses
  from cr
),
cr_feat as
(
select
cr_inc.subject_id, cr_inc.icustay_id, cr_inc.intime, cr_inc.outtime,
    cr_inc.creatinine, cr_inc.creatinine_time,
    cr_inc.age, cr_inc.inclusion_age,
    cr_inc.length_of_stay, cr_inc.inclusion_length_of_stay,
    cr_inc.rrt, cr_inc.inclusion_rrt, cr_inc.diagnoses, cr_inc.inclusion_diagnoses,
    ce.valuenum as """+features_info.loc[0,'name']+""", 
    EXTRACT('epoch' from cr_inc.creatinine_time - ce.storetime) as """+features_info.loc[0,'name']+"""_delay,
    ce.storetime as """+features_info.loc[0,'name']+"""_time,
    ce.itemid as """+features_info.loc[0,'name']+"""_itemid
  from cr_inc
  inner join """+features_info.loc[0,'table']+""" ce
    on cr_inc.subject_id = ce.subject_id
    and ce.itemid in """+ item_str +"""
    and ce."""+features_info.loc[0,'variable']+""" is not null
    and ce.storetime between cr_inc.intime and cr_inc.creatinine_time
)
select 
a.subject_id, a.icustay_id, a.intime, a.outtime,
    a.creatinine, a.creatinine_time,
    a.age, a.inclusion_age,
    a.length_of_stay, a.inclusion_length_of_stay,
    a.rrt, a.inclusion_rrt, a.diagnoses, a.inclusion_diagnoses,
    a."""+features_info.loc[0,'name']+""", a."""+features_info.loc[0,'name']+"""_delay,
    a."""+features_info.loc[0,'name']+"""_time,
    a."""+features_info.loc[0,'name']+"""_itemid
from cr_feat as a
    join (
        select creatinine_time, min("""+features_info.loc[0,'name']+"""_delay) as """+features_info.loc[0,'name']+"""_delay
        from cr_feat
        group by creatinine_time
    ) as b on a.creatinine_time = b.creatinine_time
where a."""+features_info.loc[0,'name']+"""_delay = b."""+features_info.loc[0,'name']+"""_delay
"""
df_chartevents = pd.read_sql_query(query, con)
df_chartevents = apply_inclusion_criteria(df_chartevents)

# Convert list of diagnoses into str (required to perform the merge)
df_chartevents.loc[:,'diagnoses'] = df_chartevents['diagnoses'].apply(lambda x: ', '.join(sorted(x)))

for i,row in features_info.loc[1:,:].iterrows():
    print('------------------------------------')
    print('--- Processing feature : ' + row['name'])
    # From features_info, retrieve the list of item_id to use with chartevents
    item_col = [c for c in features_info.columns.values if "item_id" in c]
    item_list = features_info.loc[i,item_col].dropna().astype('int').values
    item_str = "(" + str(item_list[0])
    for it in item_list[1:]:
        item_str = item_str + "," + str(it)
    item_str = item_str + ")"

    # List of ICD-9 codes to be excluded
    icd_list = ['5856','V420','99681'] # ESRD: 5856 / kidney transplant: V420,99681
    icd_str = "'{" + str(icd_list[0])
    for icd in icd_list[1:]:
        icd_str = icd_str + "," + str(icd)
    icd_str = icd_str + "}'"

    query = query_schema + """
    with cr1 as
    (
    select
        icu.subject_id, icu.icustay_id, icu.intime, icu.outtime, EXTRACT(EPOCH FROM icu.outtime - icu.intime)/60.0/60.0 as length_of_stay,
        ce.valuenum as creatinine, ce.storetime as creatinine_time,
        EXTRACT('epoch' from icu.intime - pat.dob) / 60.0 / 60.0 / 24.0 / 365.242 AS age,
        (rrt.icustay_id is null) as rrt,
        diag.icd9_code as diagnosis 
      from icustays icu
      inner join chartevents ce
        on icu.subject_id = ce.subject_id
        and ce.itemid = 220615
        and ce.valuenum is not null
        and ce.storetime between icu.intime and icu.outtime
      inner join patients pat
        ON icu.subject_id = pat.subject_id
      left outer join rrt 
        on icu.icustay_id = rrt.icustay_id
      inner join diagnoses_icd diag
        on icu.subject_id = diag.subject_id
        and icu.hadm_id = diag.hadm_id
    ),
    cr as
    (
    select
    cr1.subject_id, cr1.icustay_id, cr1.intime, cr1.outtime,
    cr1.creatinine, cr1.creatinine_time,
    cr1.age,
    cr1.length_of_stay,
    cr1.rrt,
    array_agg(cr1.diagnosis) as diagnoses
    from cr1
    group by cr1.subject_id, cr1.icustay_id, cr1.intime, cr1.outtime, cr1.creatinine, cr1.creatinine_time, cr1.age,
    cr1.length_of_stay,cr1.rrt
    ),
    cr_inc as
    (
    select
    cr.subject_id, cr.icustay_id, cr.intime, cr.outtime,
        cr.creatinine, cr.creatinine_time,
        cr.age, CASE
                    WHEN cr.age >= 15 then 1
                ELSE 0 END
                as inclusion_age,
      cr.length_of_stay, CASE
                            WHEN cr.length_of_stay >= 48 then 1
                         ELSE 0 END
                         as inclusion_length_of_stay,
      cr.rrt, CASE
                WHEN cr.rrt = False then 1
              ELSE 0 END
              as inclusion_rrt,
      cr.diagnoses, CASE
                WHEN cr.diagnoses && """ + icd_str + """ then 0
              ELSE 1 END
              as inclusion_diagnoses
      from cr
    ),
    cr_feat as
    (
    select
    cr_inc.subject_id, cr_inc.icustay_id, cr_inc.intime, cr_inc.outtime,
        cr_inc.creatinine, cr_inc.creatinine_time,
        cr_inc.age, cr_inc.inclusion_age,
        cr_inc.length_of_stay, cr_inc.inclusion_length_of_stay,
        cr_inc.rrt, cr_inc.inclusion_rrt, cr_inc.diagnoses, cr_inc.inclusion_diagnoses,
        ce.valuenum as """+features_info.loc[i,'name']+""", 
        EXTRACT('epoch' from cr_inc.creatinine_time - ce.storetime) as """+features_info.loc[i,'name']+"""_delay,
        ce.storetime as """+features_info.loc[i,'name']+"""_time,
        ce.itemid as """+features_info.loc[i,'name']+"""_itemid
      from cr_inc
      inner join """+features_info.loc[i,'table']+""" ce
        on cr_inc.subject_id = ce.subject_id
        and ce.itemid in """+ item_str +"""
        and ce."""+features_info.loc[i,'variable']+""" is not null
        and ce.storetime between cr_inc.intime and cr_inc.creatinine_time
    )
    select 
    a.subject_id, a.icustay_id, a.intime, a.outtime,
        a.creatinine, a.creatinine_time,
        a.age, a.inclusion_age,
        a.length_of_stay, a.inclusion_length_of_stay,
        a.rrt, a.inclusion_rrt, a.diagnoses, a.inclusion_diagnoses,
        a."""+features_info.loc[i,'name']+""", a."""+features_info.loc[i,'name']+"""_delay,
        a."""+features_info.loc[i,'name']+"""_time,
        a."""+features_info.loc[i,'name']+"""_itemid
    from cr_feat as a
        join (
            select creatinine_time, min("""+features_info.loc[i,'name']+"""_delay) as """+features_info.loc[i,'name']+"""_delay
            from cr_feat
            group by creatinine_time
        ) as b on a.creatinine_time = b.creatinine_time
    where a."""+features_info.loc[i,'name']+"""_delay = b."""+features_info.loc[i,'name']+"""_delay
    """
    
    df = pd.read_sql_query(query, con)
    df = apply_inclusion_criteria(df)
    # Convert list of diagnoses into str (required to perform the merge)
    df.loc[:,'diagnoses'] = df['diagnoses'].apply(lambda x: ', '.join(sorted(x)))
    
    df_chartevents = pd.merge(df_chartevents,df,on=['subject_id', 'icustay_id', 'intime', 'outtime', 'creatinine',
       'creatinine_time', 'age', 'length_of_stay', 'rrt', 'diagnoses'],how='outer')
    print('Merged table size : ' + str(df_chartevents.shape[0]))
    print(df_chartevents.head())



# In[6]:


# Dump to file
df_chartevents.to_csv('creatinine_measurements_1.csv')
df_chartevents.head()


# # Retrieve missing static information

# ## !!!! Add missing inclusion criteria here

# In[7]:


query = query_schema + """
with cr1 as
(
select
    icu.subject_id, icu.icustay_id, icu.intime, icu.outtime, EXTRACT(EPOCH FROM icu.outtime - icu.intime)/60.0/60.0 as length_of_stay,
    ce.valuenum as creatinine, ce.storetime as creatinine_time,
    adm.ethnicity, adm.diagnosis as diagnosis,
    pat.gender as gender,
    EXTRACT('epoch' from icu.intime - pat.dob) / 60.0 / 60.0 / 24.0 / 365.242 AS age,
    (rrt.icustay_id is null) as rrt,
    diag.icd9_code as diag 
  from icustays icu
  inner join chartevents ce
    on icu.subject_id = ce.subject_id
    and ce.itemid = 220615
    and ce.valuenum is not null
    and ce.storetime between icu.intime and icu.outtime
  inner join patients pat
    ON icu.subject_id = pat.subject_id
  inner join admissions adm
    on icu.subject_id = adm.subject_id
  left outer join rrt 
    on icu.icustay_id = rrt.icustay_id
  inner join diagnoses_icd diag
    on icu.subject_id = diag.subject_id
    and icu.hadm_id = diag.hadm_id
),
cr as
(
select
cr1.subject_id, cr1.icustay_id, cr1.intime, cr1.outtime,
cr1.creatinine, cr1.creatinine_time,
cr1.ethnicity, cr1.diagnosis,
cr1.gender as gender,
cr1.age,
cr1.length_of_stay,
cr1.rrt,
array_agg(cr1.diag) as diagnoses
from cr1
group by cr1.subject_id, cr1.icustay_id, cr1.intime, cr1.outtime, cr1.creatinine, cr1.creatinine_time,
cr1.ethnicity, cr1.diagnosis, cr1.gender, cr1.age, cr1.length_of_stay, cr1.rrt
)
select
cr.subject_id, cr.icustay_id, cr.intime, cr.outtime,
    cr.creatinine, cr.creatinine_time,
    cr.ethnicity, cr.diagnosis, cr.gender,
    cr.age, CASE
                WHEN cr.age >= 15 then 1
            ELSE 0 END
            as inclusion_age,
  cr.length_of_stay, CASE
                        WHEN cr.length_of_stay >= 48 then 1
                     ELSE 0 END
                     as inclusion_length_of_stay,
  cr.rrt, CASE
            WHEN cr.rrt = False then 1
          ELSE 0 END
          as inclusion_rrt,
  cr.diagnoses, CASE
            WHEN cr.diagnoses && """ + icd_str + """ then 0
          ELSE 1 END
          as inclusion_diagnoses
  from cr
"""
df_static = pd.read_sql_query(query, con)
df_static = apply_inclusion_criteria(df_static)

# Convert list of diagnoses into str (required to perform the merge)
df_static.loc[:,'diagnoses'] = df_static['diagnoses'].apply(lambda x: ', '.join(sorted(x)))
print(df_static.head())


# In[8]:


# Dump to file
df_static.to_csv('creatinine_measurements_2.csv')
print(df_static)


# # Merge tables into one

# In[9]:


print('Chartevents :')
print(df_chartevents.shape)
print(df_chartevents.columns.values)
print('')

print('Static:')
print(df_static.shape)
print(df_static.columns.values)


# In[10]:


merged_df = pd.merge(df_chartevents,df_static,on=['subject_id', 'icustay_id', 'intime', 'outtime', 'creatinine',
       'creatinine_time', 'age', 'length_of_stay', 'rrt', 'diagnoses'],how='outer')


# # Remove columns that are not features (except icustay_id and patient_id that are needed to build the table with labels)

# In[11]:


# REMOVE COLUMNS THAT WERE USED FOR INCLUSION CRITERIA BUT THAT ARE NOT AVAILABLE FEATURES FOR THE PREDICTION
# REMOVE ICUSTAY_IDS FOR WHICH THERE IS ONLY ONE MEASUREMENT OF CREATININE

print('Number of lines : ' + str(merged_df.shape[0]))
to_remove = ['intime','outtime','length_of_stay','rrt','diagnoses']
for c in to_remove:
    if (c in merged_df.columns.values): merged_df = merged_df.drop(c,axis=1)
        
# Remove columns with _time suffix EXCEPT the time for creatinine which is required to compute the labels
to_remove = [i for i in merged_df.columns.values if (('_time' in i) & (i!='creatinine_time'))]
for c in to_remove:
    if (c in merged_df.columns.values): merged_df = merged_df.drop(c,axis=1)

# Remove columns with _itemid suffix
to_remove = [i for i in merged_df.columns.values if '_itemid' in i]
for c in to_remove:
    if (c in merged_df.columns.values): merged_df = merged_df.drop(c,axis=1)

# Remove icustay_ids for which there's only one measurement of creatinine
count_mes = merged_df['icustay_id'].value_counts()
to_remove = count_mes.index.values[count_mes==1]
for i in to_remove:
    merged_df = merged_df.loc[merged_df['icustay_id']!=i,:]

# Drop duplicate values of creatinine_time
merged_df = merged_df.drop_duplicates(['icustay_id','creatinine_time'])
print('After dropping unique measurements of creatinine : ' + str(merged_df.shape[0]))
merged_df.head()


# **WARNING : the columns named "..._itemid" can be used to filter on the itemid used to retrieve the feature after the query has been done. But these are not features.**

# In[12]:


# Dump to file
merged_df.to_csv('creatinine_measurements_merged.csv')
print(merged_df)


# # Create labels for evolution of creatinine

# In[13]:


merged_df = pd.read_csv('creatinine_measurements_merged.csv').drop('Unnamed: 0',axis=1)


# In[14]:


# Keep only creatinine measured between 0am and 5am
merged_df.loc[:,'creatinine_hour'] = merged_df['creatinine_time'].astype('datetime64[ns]').apply(lambda x: x.hour)
merged_df = merged_df.loc[((merged_df['creatinine_hour']>=0) & (merged_df['creatinine_hour']<=5)) ,:]


# In[15]:


# In each row, report values of creatinine for next day and days before
merged_df.loc[:,'creatinine_time'] = merged_df.loc[:,'creatinine_time'].astype('datetime64[ns]')
for i, row in merged_df.iterrows():
    same_stay = merged_df.loc[merged_df['icustay_id']==row['icustay_id'],:]
    delay = (same_stay['creatinine_time']-row['creatinine_time']).apply(lambda x: x.days)
    merged_df.loc[i,'creatinine_tomorrow'] =same_stay.loc[delay==1,'creatinine'].max()
    merged_df.loc[i,'creatinine_yesterday'] =same_stay.loc[delay==-1,'creatinine'].max()
    merged_df.loc[i,'creatinine_before_yesterday'] =same_stay.loc[delay==-2,'creatinine'].max()


# In[16]:


def create_labels(diff):
    if diff>0.15 :
        return 0
    elif diff<-0.15 :
        return 1
    elif math.isnan(diff):
        return None
    else:
        return 2

merged_df.loc[:,'creatinine_diff'] = merged_df['creatinine_tomorrow']-merged_df['creatinine']
merged_df.loc[:,'label'] = merged_df['creatinine_diff'].apply(create_labels)

# Drop NaN labels (lines with no "creatinine_tomorrow")
merged_df = merged_df.dropna(subset=['label'])
print(merged_df)


# In[17]:


print(merged_df)


# In[18]:


# Remove features that are not labels
to_remove = ['creatinine_time','creatinine_hour','creatinine_tomorrow','creatinine_diff']
for c in to_remove:
    if (c in merged_df.columns.values): merged_df = merged_df.drop(c,axis=1)
print(merged_df.columns)


# In[19]:


# Dump to file
merged_df.to_csv('dataset_with_labels.csv')
print(merged_df)

