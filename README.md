# Data Science Challenge

## Predict time to credit default for a customer basis the variables provided across various Categories.

Top 5 among 3600 participants in the public leaderboard among students from all IIT Campuses and Economic Institutes. (Team: Dopa-mean)

### Overview: 

![Model Outline](https://github.com/Wickkey/American-Express-Campus-Superbowl-Challenge-2022/blob/main/Images/Model_Overview.png?raw=true)


## Files Present:
#### - Notebooks Folder:
1) Denoiser.ipynb is notebook relevant to one-hot encoding categorical variables and denoising the data
2) Meta_features_level1.ipynb is notebook relevant to adding layer1 meta features 
3) Meta_features_level2.ipynb is notebook relevant to adding layer2 meta features
4) Ensembling.ipynb is notebook relevant to voting 4 classifiers

1,2,3,4 deals with the methodology, training



Folders "Classifier Models" and "Meta Models" contains pickle files related to trained models


#### - Generate Predictions:
5) To get predictions on any data, 
- Replace input_file_name and output_file_name appropriately and 
- Run Generate_Predictions.py file
