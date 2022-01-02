# Making Emojis More Predictable

by **Karan Abrol**, **Karanjot Singh** and **Pritish Wadhwa**, Natural Language Processing (CSE546) under the guidance of **Dr. Shad Akhtar** from **Indraprastha Institute of Information Technology, Delhi**.

## Introduction
<p align="justify">The advent of social media platforms like WhatsApp, Facebook (Meta) and Twitter, etc. has changed natural language conversations forever. Emojis are small ideograms depicting objects, people, and scenes (Cappallo et al., 2015). Emojis are used to complement short text messages with a visual enhancement and have become a de-facto standard for online communication. Our aim is to predict a single emoji that appears in the input tweets. </p>  
<p align="justify">In this project, we aim to achieve the task of predicting emojis from tweets. We aim to investigate the relationship between words and emojis.</p>

## Project Pipeline Summary
<p align="justify">
We started off by collecting the data. The data was then thoroughly studied and preprocessed. Key features were also extracted at this stage. Due to computational restrictions, a subset of data was taken which was further divided into training, test- ing and validation split, such that the distribution of any class in any two sets were same. After this, various machine learning and deep learning models were applied on the data set and the results were generated and analysed.
</p>

## Deployment
![Emoji Prediction Website](https://fierce-garden-64530.herokuapp.com)
### Screenshots
!Prediction Website1](https://user-images.githubusercontent.com/55680995/147876786-56302f39-740f-4b13-afca-420245b7fa53.jpg)
![Prediction Website2](https://user-images.githubusercontent.com/55680995/147876808-a86083dd-20fd-4b6a-92b0-84fbad23ceee.jpg)



## Dataset
<p align="justify">
The data we used consists of a list of tweets associated with a single emoji, indexed by 20 labels for each of the 20 emojis. 5,00,000 Tweets by users in the United States, from October 2015 to Jan 2018, were retrieved using the Twitter API. The script for scraping this dataset was made available by the SemEval 2018 challenge. Due to computational limitations we merged the test and trial data, and further divided that into training, trial and test data with a split of 70:10:20. We maintained the label ratios for each emoji across the three sets to best reflect how frequently they are used in real life.
</p>

## Models
- Machine Learning Models:
  - Logistic Regression
  - K-Nearest Neighbours
  - Stochastic Gradient Descent
  - Random Forest Classifier
  - Naive Bayes
  - Adaboost Classifier
  - Support Vector Machine
  - 

- Deep Learning Models:
  - RNN
  - LSTM
  - BiLSTM

<!-- ## Repository Description
- ### Preprocessing
  Code files for preprocessing data, EDA, feature selection, encoding, train-val-test split and feature scaling
- ### Regression
  Code files for training, validating, generating graphs and saving regression models
- ### Classification
  Code files for training, validating, generating graphs and saving classification models
- ### Reports
  Proposal, Interim Report and Final Report
- ### Images
  Images used in reports -->

## Contact
For further queries feel free to reach out to following contributors.  
Karan Abrol (karan19366@iiitd.ac.in)  
Karanjot Singh (karanjot19050@iiitd.ac.in)  
Pritish Wadhwa (pritish19440@iiitd.ac.in)

## Final Report
![Final Report 1](/Reports/Final_Report_Images/Report-1.png)  
![Final Report 2](/Reports/Final_Report_Images/Report-2.png)  
![Final Report 3](/Reports/Final_Report_Images/Report-3.png)  
![Final Report 4](/Reports/Final_Report_Images/Report-4.png)  
![Final Report 5](/Reports/Final_Report_Images/Report-5.png)  
![Final Report 6](/Reports/Final_Report_Images/Report-6.png)  
![Final Report 7](/Reports/Final_Report_Images/Report-7.png)
