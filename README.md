# Hackathon

This repository gathers all pieces of codes produced by the crea_forecast team during the [Datathon for Intensive Care](http://blogs.aphp.fr/dat-icu/) that took place in Paris on January 20-21st, 2018. 

## Objective
This is a POC of using data from intensive care units in order to predict if a patient's kidney health will rather improve or worsen within the next 24 hours. The creatinine rates in blood will be used as an indicator of the kidney's condition.

The aim is to build a classification model that allows to predict the most probable among three classes :
- The creatinine will increase in the next ≈24hours (i.e. the patient's condition will worsen)
- The creatinine will decrease in the next ≈24hours 
- The creatinine will remain stable within next ≈24hours

## Dataset
The dataset is built from the [MIMIC-III dataset](https://mimic.physionet.org/) [1].

## Prerequisites

### Build database
- Request access to the MIMIC-III dataset : https://mimic.physionet.org/gettingstarted/access/
- Download the MIMIC-III GitHub repository : https://github.com/MIT-LCP/mimic-code/
- Follow instructions to build the SQL database : https://github.com/MIT-LCP/mimic-code/tree/master/buildmimic/postgres
- Build an additional table. This will create a psql table named rrt : 
```cd mimic-code/concepts/ 
psql -U postgres -d mimic -a -f rrt.sql
```

### Dependencies
- The source code is written in Python 3.
- The python packages can be installed with pip : `pip3 install -R requirements.txt`
- To use Keras models, first install tensorflow : https://www.tensorflow.org/install/
- WARNING : XGBoost installation with pip is currently disabled for Windows. Instructions for Windows users : https://xgboost.readthedocs.io/en/latest/build.html

## Usage
### Make_dataset.ipynb 
**Extracts the dataset from the SQL database**
- Input file : features_info.csv
- Output file : dataset_with_labels.csv (not hosted on this repository, you have to create it yourself)

### Explore_dataset.ipynb
**Automates some computations of basic statistics and plots for each feature in the dataset**
- Input file : dataset_with_labels.csv

### Train_models.ipynb
**Trains some multiclass classifiers from different packages (scikit-learn, keras, XGBoost) and compare their performances**
- Input file : dataset_with_labels.csv

## Team contributors
R. Barthélémy
A. Beinrucker
L. Cetinsoy
B. Chousterman
S. Falini
M. Jamme
M. Kovanis
A. Mutschler
M. Naeem

## References
[1] MIMIC-III, a freely accessible critical care database. Johnson AEW, Pollard TJ, Shen L, Lehman L, Feng M, Ghassemi M, Moody B, Szolovits P, Celi LA, and Mark RG. Scientific Data (2016). DOI: 10.1038/sdata.2016.35. 
