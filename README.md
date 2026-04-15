mtsamples.csv is the partially cleaned dataset.

preprocessing.py finishes preprocessing mtsamples.csv and outputs data1.csv

eda.py does the exploratory data analysis on data1.csv

feature_extraction.py takes the preprocessed dataset and returns x labels for both TF-IDF
and ClinicalBERT. It also returns the y labels.

analysis.py does logistic regression on the x,y labels obtained from the feature extraction.