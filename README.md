# Project-one
# Topic Selected:  Used Cars Price Prediction

## Background:

Over the past few years, the automotive industry has faced a shortage in the Semiconductor Integrate Chips globally. The Semicoductor IC is a critical component for controlling several electronic devices in the vehicle. 
Even though the car industry is growing at a fast rate, the shortage is impeding the growth of new car production and sales. The new car sales industry is making up for the hsortage by raising their APR and prices. Therefore, there is a trend of increasing demand of used cars which is making the prices of used cars higher as well.
Based on the current situation in the automotive industry, we have decided on predicting the prices of used cars for our project. Though it is a global issue, we will limit our studies and findings for US market only.


## Data Sources:
https://www.kaggle.com/code/maciejautuch/car-price-prediction/data
This dataset collected in Kaggle is mainly from craiglist.org (used item selling website) from all over US. These cars are from different manufacturers and of different years.
--------(we can add data image here)-------------------

## Required Libraries:
List of all necessary liblaries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import plotly.express as px

import seaborn as sns

import pandas_profiling as pp

### Libraries for preprocessing


from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import PolynomialFeatures

### Liblaries for models

from sklearn.linear_model import LinearRegression, Ridge

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, BaggingRegressor

### Libraries for cross validation and model evaluation


from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


-------Here is the list of libraries we will be needing (may change):

*	Panda
*	Sql
*	Matplotlib
*	Sklearn
*	Tableau
*	Scikit-learn
*	Random forest
*	Etc---------------

## Questions to Answer:

1.	How does the mileage affect the price of the used car?
2.	How does size of the car impact the price?
3.	How does the age of the car, condition and fuel type affect the price of the car?
4.	Will this affect the overall demand for a used car in place of a new car for consumers?

## Steps to be taken:
Data Cleaning:
 that includes getting rid of all undesired columns
 
Creating table in SQL

Utilize Tableau 

Machine Learning Model:
this includes choosing **X** variable as a **collection of features** and **Y** as a target variable which will be **Price**

Model for Regression

Linear Regression
Logistic Regression
K&N Algorithm
Decission Tree	
etc
## Group Details:

We are team of four people. 

Shahla and Samantha are in charge of gathering information about the results of the dataset and what we want to achieve. Ryiochi was in charge of cleaning up the csv and dataset. Matthew was initializing our databases.

