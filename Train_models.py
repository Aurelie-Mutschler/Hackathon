
# coding: utf-8

# # Prediction of the evolution of creatinine rates

# ## 0. Load useful modules

# In[1]:


import numpy as np
from pandas import pandas as pd
#import time
#import copy
#from pprint import pprint

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier

from keras.models import Sequential
from keras.layers import Dense, Dropout
#from keras.callbacks import EarlyStopping
#from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier

from xgboost import XGBClassifier

#from sklearn.model_selection import cross_val_score, train_test_split, StratifiedShuffleSplit, GridSearchCV
#from sklearn.metrics import fbeta_score, make_scorer, classification_report, confusion_matrix
#from sklearn.ensemble import AdaBoostClassifier
#from xgboost import XGBClassifier

from termcolor import colored, cprint


# In[2]:


# fix random seed for reproducibility
# THIS MUST BE CALLED IN EACH CELL WHERE RANDOM NUMBER GENERATORS ARE USED TO ENSURE REPRODUCIBLITY
# Even in the case where we re-run a given cell
seed=44
np.random.seed(seed)


# ## 1. Data preprocessing

# ### Load dataset

# In[3]:


# Load dataset
data = pd.read_csv("dataset_with_labels.csv", engine='python').drop('Unnamed: 0',axis=1).reset_index(drop=True)
print('Initial number of examples : ', data.shape[0])
print('Initial number of columns : ', data.shape[1])
print()

# Remove columns that are ids
id_cols = [c for c in data.columns.values if ('_id' in c)]
print('Dropping ' + str(len(id_cols)) + ' columns that are not features : ')
print(id_cols)
for c in id_cols:
    data = data.drop(c,axis=1)
print('After dropping ids, number of columns : ', data.shape[1])
data.head()


# ### Handle missing values

# In[4]:


# Throw features that have a rate of missing values > x% 
max_missing_rate = 0.3
missing_rates = data.apply(lambda x : x.isnull(), axis=1).sum(axis=0)/data.shape[0]
to_drop = missing_rates[missing_rates>0.3].index.values

print('Following columns have missing rate >' + str(int(100*max_missing_rate)) + '% and will be dropped : ')
print(to_drop)
data = data.drop(labels=to_drop,axis=1)
print('Number of columns kept : ', data.shape[1])
print(data.columns.values)
print()

# Then drop lines where there remain some missing values
print('Drop lines where there are some NAs remaining...')
data = data.dropna(axis=0,how='any').reset_index(drop=True)
print('Number of lines kept : ', data.shape[0])
print()

# Check there is not missing value left in dataset
print('Number of missing values remaining : ', data.isnull().sum().sum())


# ### Encode categorical variables

# In[5]:


# Before performing dummy encoding, map some categorical variables to fewer modalities
# This is to control the number of features
print('Mapping modalities of ethnicity towards simplified modalities...')
simple_ethnicity = {
   'BLACK/AFRICAN AMERICAN': 'BLACK', 
   'WHITE': 'WHITE', 
   'UNKNOWN/NOT SPECIFIED': 'UNKNOWN',
   'HISPANIC/LATINO - DOMINICAN': 'OTHER', 
   'UNABLE TO OBTAIN': 'UNKNOWN',
   'PATIENT DECLINED TO ANSWER': 'UNKNOWN', 
   'ASIAN - CHINESE': 'ASIAN',
   'AMERICAN INDIAN/ALASKA NATIVE': 'OTHER', 
   'MULTI RACE ETHNICITY': 'OTHER',
   'WHITE - OTHER EUROPEAN': 'WHITE', 
   'OTHER': 'OTHER', 
   'PORTUGUESE': 'WHITE',
   'HISPANIC OR LATINO': 'OTHER', 
   'ASIAN': 'ASIAN', 
   'HISPANIC/LATINO - PUERTO RICAN': 'OTHER',
   'MIDDLE EASTERN': 'OTHER', 
   'ASIAN - KOREAN': 'ASIAN', 
   'BLACK/HAITIAN': 'BLACK',
   'ASIAN - OTHER': 'ASIAN', 
   'HISPANIC/LATINO - CUBAN': 'OTHER', 
   'ASIAN - FILIPINO': 'ASIAN',
   'BLACK/CAPE VERDEAN': 'BLACK', 
   'WHITE - BRAZILIAN': 'WHITE', 
   'ASIAN - ASIAN INDIAN': 'ASIAN',
   'WHITE - EASTERN EUROPEAN': 'WHITE', 
   'HISPANIC/LATINO - GUATEMALAN': 'OTHER',
   'ASIAN - VIETNAMESE': 'ASIAN', 
   'HISPANIC/LATINO - MEXICAN': 'OTHER',
   'WHITE - RUSSIAN': 'WHITE', 
   'BLACK/AFRICAN': 'BLACK', 
   'ASIAN - CAMBODIAN': 'ASIAN',
   'AMERICAN INDIAN/ALASKA NATIVE FEDERALLY RECOGNIZED TRIBE': 'OTHER'
}

data.loc[:,'ethnicity'] = data['ethnicity'].map(simple_ethnicity)
print('Remaining values for ethnicity : ')
print(data['ethnicity'].value_counts(dropna=False))
print()

print('Mapping modalities of diagnosis towards simplified modalities...')
to_sepsis = data['diagnosis'].apply(lambda x : 'FEVER' in x)
data.loc[to_sepsis,'diagnosis'] = 'SEPSIS'

to_resp_failure = data['diagnosis'].apply(lambda x : (('DYSPNEA' in x) | ('SHORTNESS OF BREATH' in x)))
data.loc[to_resp_failure,'diagnosis'] = 'RESPIRATORY FAILURE'

diag_to_keep = ['PNEUMONIA', 'CONGESTIVE HEART FAILURE', 'SUBARACHNOID HEMORRHAGE',
              'INTRACRANIAL HEMORRHAGE', 'ALTERED MENTAL STATUS', 'CORONARY ARTERY DISEASE',
              'ABDOMINAL PAIN', 'CHEST PAIN', 'HYPOTENSION', 'ACUTE RENAL FAILURE',
              'RESPIRATORY FAILURE', 'GASTROINTESTINAL BLEED', 'PANCREATITIS', 'SEPSIS']
to_other = data['diagnosis'].apply(lambda x : x not in diag_to_keep)
data.loc[to_other, 'diagnosis'] = 'OTHER'

print('Remaining values of diagnosis : ')
print(data['diagnosis'].value_counts(dropna=False))


# In[6]:


# Perform dummy encoding
dummy_variables = ['ethnicity','diagnosis','gender']
print('Performing dummy encoding for features : ')
print(dummy_variables)
data = pd.get_dummies(data,columns=dummy_variables)
print('New list of columns :')
print(data.columns.values)


# ### Remove outliers

# In[7]:


feature_ranges = {
    'creatinine': {'min': 0.0, 'max': 20.0},
    'creatinine_yesterday': {'min': 0.0, 'max': 20.0},
    'creatinine_before_yesterday': {'min': 0.0, 'max': 20.0},
    'potassium': {'min': 1.2, 'max': 7.0},
    'ph_blood': {'min': 0.0, 'max': 14.0},
    'age': {'min': 0.0, 'max': 110.0},
    'bilirubin': {'min': 0.0, 'max': 20.0}
}

print('Initial number of examples : ', data.shape[0])
for k in feature_ranges:
    if k in data.columns.values:
        print('Dropping outliers for feature : ', k)
        data = data.loc[((data[k]>feature_ranges[k]['min']) & (data[k]<feature_ranges[k]['max'])),:]
        print('Remaining number of examples : ', data.shape[0])


# ### Normalize data

# In[8]:


# Create a sklearn StandardScaler() for all numeric variables
numeric_variables = ['creatinine', 'age', 'arterial_pressure_systolic', 'arterial_pressure_diastolic',
                     'heart_rate', 'temperature', 'ph_blood']
print('Standardizing following features : ', numeric_variables)
numeric_scaler = StandardScaler()
data.loc[:,numeric_variables] = numeric_scaler.fit_transform(data.loc[:,numeric_variables])
for c,col in (data.loc[:,numeric_variables].describe(percentiles=[]).loc[['mean','std'],:]).iteritems():
    print('--- Feature : ', c)
    print(col)
print()

# Special case for delays : as they are all times in seconds, create one unique scaler for all delays
# (i.e. apply same transformation to each time measurement, such that they remain comparable)
time_variables = [i for i in data.columns.values if '_delay' in i]
print('Standardizing following features : ', time_variables)
time_scaler = StandardScaler()
time_scaler.fit(data.loc[:,time_variables].values.flatten())
data.loc[:,time_variables] = time_scaler.transform(data.loc[:,time_variables])

for c,col in (data.loc[:,time_variables].describe(percentiles=[]).loc[['mean','std'],:]).iteritems():
    print('--- Feature : ', c)
    print(col)
print()


# ### Prepare X and y input datasets
# Split dataset into three sets :
# - **hyperopt set (80%)** : for each model, perform grid search with cross-validation for hyperparameter optimization.
# - **compare set (10%)** : compare performances obtained on this set for each model (with optimized hyperparameters)
# - **test set (10%)** : in the end, check for overfitting by comparing the change in performances between compare_set and test_set

# In[9]:


# THIS MUST BE CALLED IN EACH CELL WHERE RANDOM NUMBER GENERATORS ARE USED TO ENSURE REPRODUCIBLITY
# Even in the case where we re-run a given cell
np.random.seed(seed)

# Separate label from features
y = data['label']
X = data.drop('label',axis=1)

# Split into hyperopt, compare and test sets
X_hyperopt, X1, y_hyperopt, y1 = train_test_split(X, y, train_size=0.8, stratify=y.values)
X_compare, X_test, y_compare, y_test = train_test_split(X1, y1, train_size=0.5, stratify=y1.values)

print('--- Hyperopt set :')
print('Number of examples : ', X_hyperopt.shape[0])
print('Class balance : ')
print(y_hyperopt.value_counts(dropna=False))
print('--- Compare set :')
print('Number of examples : ', X_compare.shape[0])
print('Class balance : ')
print(y_compare.value_counts(dropna=False))
print('--- Test set :')
print('Number of examples : ', X_test.shape[0])
print('Class balance : ')
print(y_test.value_counts(dropna=False))


# ## 2. Hyperparameter tuning

# ### Initialize models and their respective set of hyperparameters for grid search
# - Create list of dict : [{**'model_name'**: sklearn_model, **'hyperparam_grid'**:{**'hyperparam_name'**:[list of possible values]}}]
# - then just loop on this list to perform hyperparameter optim for each model

# In[10]:


# Define a functions for building keras models with different architectures

def build_1Lperceptron(loss, optimizer, metrics, input_shape, hl_1_n_units, hl_1_activation,
                      hl_1_dropout_rate, ol_n_units, ol_activation):
    # Build the model architecture 
    model = Sequential()
    model.add(Dense(units=hl_1_n_units, activation=hl_1_activation,
                    input_shape=input_shape))
    if (hl_1_dropout_rate != 0.0): model.add(Dropout(hl_1_dropout_rate))
    
    model.add(Dense(units=ol_n_units, activation=ol_activation))
              
    # Compile the model using a loss function and an optimizer.
    model.compile(loss = loss, optimizer=optimizer, metrics=metrics)
    return model

def build_2Lperceptron(loss, optimizer, metrics, input_shape, hl_1_n_units, hl_1_activation,
                      hl_1_dropout_rate, hl_2_n_units, hl_2_activation,
                      hl_2_dropout_rate, ol_n_units, ol_activation):
    # Build the model architecture 
    model = Sequential()
    model.add(Dense(units=hl_1_n_units, activation=hl_1_activation,
                    input_shape=input_shape))
    if (hl_1_dropout_rate != 0.0): model.add(Dropout(hl_1_dropout_rate))
    
    model.add(Dense(units=hl_2_n_units, activation=hl_2_activation))
    if (hl_2_dropout_rate != 0.0): model.add(Dropout(hl_2_dropout_rate))
                 
    model.add(Dense(units=ol_n_units, activation=ol_activation))
              
    # Compile the model using a loss function and an optimizer.
    model.compile(loss = loss, optimizer=optimizer, metrics=metrics)
    return model


# In[11]:


# THIS MUST BE CALLED IN EACH CELL WHERE RANDOM NUMBER GENERATORS ARE USED TO ENSURE REPRODUCIBLITY
# Even in the case where we re-run a given cell
np.random.seed(seed)

# Classifiers and their respective grids for hyperparameter tuning should be specified here :
models = {
    'linear_SVM': {
        'classifier': OneVsRestClassifier(SGDClassifier(loss='hinge', penalty='l2', learning_rate='optimal', 
                                                        class_weight='balanced'), n_jobs=-1),
        'search_grid': {
            'estimator__alpha': [0.000001,0.000003,0.00001,0.00003,0.0001,0.0003,0.001, 0.003, 0.1, 0.3, 1.0]   
        }
    },
    'logreg': {
        'classifier': OneVsRestClassifier(SGDClassifier(loss='log', penalty='l2', learning_rate='optimal', 
                                                        class_weight='balanced'), n_jobs=-1),
        'search_grid': {
            'estimator__alpha': [0.000001,0.000003,0.00001,0.00003,0.0001,0.0003,0.001, 0.003, 0.1, 0.3, 1.0]
        }
    },
    'logreg_elasticnet': {
        'classifier': OneVsRestClassifier(SGDClassifier(loss='log', penalty='elasticnet', learning_rate='optimal',
                                                        class_weight='balanced'), n_jobs=-1),
        'search_grid': {
            'estimator__alpha': [0.000001,0.000003,0.00001,0.00005,0.0001,0.0005,0.001],
            'estimator__l1_ratio': [0.15,0.3,0.6,1.0]
        }
    }
#    'xgboost': {
#        'classifier': XGBClassifier(eval_metric='mlogloss', num_class= 3, objective= 'multi:softmax', class_weight='balanced'),
#        'search_grid': {
#            'learning_rate': [0.003,0.01,0.03,0.1,0.3,1.0,3.0],
#            'min_child_weight': [0.1,0.3,1,3],
#            'max_depth': [3,6,8,10,12]
#        }
#    },
#        'perceptron_1hl': {
#        'classifier': KerasClassifier(build_fn=build_1Lperceptron, epochs=10,batch_size=10),
#        'search_grid': {
#            'loss': ['categorical_crossentropy'],
#            'optimizer': ['adadelta'],
#            'metrics':[['accuracy']],
#            'input_shape': [(X_hyperopt.shape[1],)],
#            'hl_1_n_units': [10,20,30,40],
#            'hl_1_activation': ['relu'],
#            'hl_1_dropout_rate': [0.0,0.1,0.3,0.5],
#            'ol_n_units': [3],
#            'ol_activation': ['softmax']
#        }
#    },
#    'perceptron_2hl': {
#        'classifier': KerasClassifier(build_fn=build_2Lperceptron, epochs=10,batch_size=10),
#        'search_grid': {
#            'loss': ['categorical_crossentropy'],
#            'optimizer': ['adadelta'],
#            'metrics':[['accuracy']],
#            'input_shape': [(X_hyperopt.shape[1],)],
#            'hl_1_n_units': [10,20,40],
#            'hl_1_activation': ['relu'],
#           'hl_1_dropout_rate': [0.0,0.2],
#            'hl_2_n_units': [10,20,40],
#            'hl_2_activation': ['relu'],
#            'hl_2_dropout_rate': [0.0,0.2],
#            'ol_n_units': [3],
#            'ol_activation': ['softmax']
#        }
#    }
#    'random_forest': {
#        'classifier': RandomForestClassifier(n_estimators=10, max_depth=None),
#        'search_grid': {
#            'n_estimators': [int(x) for x in np.linspace(200,2000,10)],
#            'max_depth': np.append([int(x) for x in np.linspace(10,100,10)], None)
#        }
#    }
}


# ### Grid search for each model
# /!\ this is time-consuming, pay attention to save the intermediate results of the grid search in a file
# - in the end, create a list containing the best classifier obtained for each model : [sklearn_model(best hyperparams), kerasclassifier(best_hyperparams),...]

# In[12]:


# Create custom performance metrics that will be used for comparing performances accross different models
def sensitivity_increase(y,y_pred):
    conf_mat = confusion_matrix(y, y_pred)
    return conf_mat[0,0]/(conf_mat[0,0] + conf_mat[0,1] + conf_mat[0,2])

def specificity_decrease(y,y_pred):
    conf_mat = confusion_matrix(y, y_pred)
    return (conf_mat[0,0] + conf_mat[0,2] + conf_mat[2,0] + conf_mat[2,2])/(conf_mat[0,0] + conf_mat[0,2] 
                                                                            + conf_mat[2,0] + conf_mat[2,2]
                                                                           + conf_mat[0,1] + conf_mat[2,1])

# The dictionnary below can be used in GridSearchCV() for multiple scoring
# /!\ This is doable only with sklearn 0.19 (current stable release: 0.18)
# For now we use only sensitivity_increase for hyperparameter optimization
scores = {
    'accuracy': accuracy_score,
    'sensitivity_increase': sensitivity_increase, 
    'specificity_decrease': specificity_decrease
}


# In[13]:


# THIS MUST BE CALLED IN EACH CELL WHERE RANDOM NUMBER GENERATORS ARE USED TO ENSURE REPRODUCIBLITY
# Even in the case where we re-run a given cell
np.random.seed(seed)

# Loop over each model and perform grid search to tune its hyperparameters
for i in models.keys():
    print('---- Grid search for ' + str(i) + ' ---')
    clf = models[i]['classifier']
    params = models[i]['search_grid']
    
    #gridsearch = GridSearchCV(estimator=clf, param_grid=params, refit=True, verbose=1)
    gridsearch = GridSearchCV(estimator=clf, param_grid=params, scoring='accuracy', 
                              refit=True, verbose=1)
    gridsearch.fit(X_hyperopt,y_hyperopt)
    models[i]['best_classifier'] = gridsearch.best_estimator_
    models[i]['best_scores'] = gridsearch.best_score_
    models[i]['best_params'] = gridsearch.best_params_
    print('Best score : ', gridsearch.best_score_)
    print('Best set of hyperparameters : ', gridsearch.best_params_)


# ## 3. Comparison of performances
# Re-train models with their respective optimal sets of hyperparameters, on ALL the "hyperopt" dataset. Then assess and compare performances on the "compare" dataset

# In[14]:


# THIS MUST BE CALLED IN EACH CELL WHERE RANDOM NUMBER GENERATORS ARE USED TO ENSURE REPRODUCIBLITY
# Even in the case where we re-run a given cell
np.random.seed(seed)

compare_performances = pd.DataFrame(index=list(models.keys()),columns=list(scores.keys()))
for k in list(models.keys()):
    clf = models[k]['best_classifier']
    # Use ALL examples from "hyperopt" dataset to re-train model
    # NB: this is not mandatory as GridSearchCV(refit=True) does it.
    clf.fit(X_hyperopt,y_hyperopt)
    # Assess performances on "compare" dataset
    y_pred = clf.predict(X_compare)
    for m in list(scores.keys()):
        compare_performances.loc[k,m] = scores[m](y_compare, y_pred)
print(compare_performances)


# ## 4.  Check for overfitting
# Re-train models on ALL the examples from "hyperopt" and "compare", then assess performances on the "test" dataset.
# NB : if there's a significant discrepancy in performances between "compare" and "test" datasets, we may be overfitting or suffering from bias. The generalized performances are the ones obtained on the "test" set.

# In[15]:


# THIS MUST BE CALLED IN EACH CELL WHERE RANDOM NUMBER GENERATORS ARE USED TO ENSURE REPRODUCIBLITY
# Even in the case where we re-run a given cell
np.random.seed(seed)

test_performances = pd.DataFrame(index=list(models.keys()),columns=list(scores.keys()))
for k in list(models.keys()):
    clf = models[k]['best_classifier']
    # Use ALL examples from "hyperopt" AND "compare" datasets to re-train models
    clf.fit(X_hyperopt.append(X_compare),y_hyperopt.append(y_compare))
    # Assess performances on "test" dataset
    y_pred = clf.predict(X_test)   
    for m in list(scores.keys()):
        test_performances.loc[k,m] = scores[m](y_test, y_pred)
        
print('Performances on "compare" set : ')
print(compare_performances)
print()
print('Performances on "test" set (= generalized) : ')
print(test_performances)


# ## 5. Alternative to be tested : Don't drop any missing values from the dataset but rather train an XGBoost classifier that handles missing values
