### Data Analytics Bootcamp Final Project
# Used Cars Price Prediction

## Deliverables of the project
- Code necessary for data exploration and analysis: *** FILE NAME ***
- Python code for data preprocessing and machine learning: *** FILE NAME ***
- Dashboard: *** LINK TO DASHBOARD ***
- Presentation slide: *** LINK TO GOOGLE SLIDE ***


### INDEX
#### I.		Background
#### II.		Object
#### III.		Data Sources
#### IV.	Questions to Answer
#### V.		Technologies, Languages, Tools, and Algorithms Used throughout Project
#### VI.	Workflow of Project


## I. Background
Over the past few years, the automotive industry has faced a shortage in the Semiconductor Integrate Chips globally. The Semicoductor IC is a critical component for controlling several electronic devices in the vehicle. 
Even though the car industry is growing at a fast rate, the shortage is impeding the growth of new car production and sales. The new car sales industry is making up for the shortage by raising their APR (annual percentage rate) and prices. Therefore, there is a trend of increasing demand of used cars which is making the prices of used cars higher as well.

## II. Objective
Based on the current situation in the automotive industry, we have decided on predicting the prices of used cars for our project using the set of variables in a dataset. Though it is a global issue, we will limit our studies and findings for US market only.

## III. Data Sources
https://www.kaggle.com/code/maciejautuch/car-price-prediction/data
This dataset collected in Kaggle.com is mainly from craigslist.org (used item selling website) from all over the US. The cars were made by different manufacturers, and the years of entry varied widely.

## IV. Questions to Answer
1.	How does the mileage affect the price of the used car?
2.	How does size of the car impact the price?
3.	How does the features such as entry year of the car, condition and fuel type affect the price of the car?

## V. Technologies, Languages, Tools, and Algorithms Used throughout Project
* #### Python
Python is a computer programming language often used to build websites and software, automate tasks, and conduct data analysis. Python is a general-purpose language, which means that it can be used to create a variety of different programs and isn't specialized for any specific problem. Python library provides a collection or package of various modules.
* #### Pandas
Pandas is a software library written for the Python programming language. It is a fast, powerful, flexible, and easy-to-use open-source for data analysis and manipulation.
* #### Scikit-Learn (Sklearn)
	Scikit-Learn is the most useful library for machine learning in Python, which provides a selection of efficient tools for machine learning and statistical modeling. Machine learning models are the mathematical engines of artificial intelligence, expressions of algorithms that find patterns and make predictions faster than a human can.
* #### Postgres pgAdmin (SQL)
PostgreSQL, also known as Postgres, is an open-source relational database with a strong reputation for its reliability, flexibility and support of open technical standards. PostgreSQL supports both non-relational and relational data types. pgAdmin is the community client for using PostgreSQL. It is usually installed along with PostgreSQL. While psql is a simple command-line tool, pgAdmin is a graphical user interface that provides pretty much the same functionality.
* #### Tableau
Tableau software is a tool that helps visualize big data. The main use of Tableau software is to help people see and understand their data through its interactive features.
* #### Google Slide
Google Slides is an online presentation application that lets users create and format presentations while working with others.

## VI. Workflow of Project

### 1. Data exploration
We planned to work with pandas in the Jupyter Notebook. The original dataset, which was downloaded in a csv format from the Kaggle.com site as indicated in III. Data Sources, was imported into the Jupyter Notebook.
#### a. Summary of data
	The original data contains 426,880 rows and 26 columns. The summary of the data is shown in Table 1. As the objective of the project is to predict the used car price, the price will be assigned as a target variable, while the other 25 columns will pass as features. Also, we will be dropping off null values during further exploration and analysis. We cannot feed the raw data in machine learning models, so we worked on various cleaning of the data. 

#### (Table 1)
 <img src="INCLUDE LINK HERE" width="320"/>  

#### b. Selection of relevant columns
	Based on review of the data summary, we decided to select the following columns as they seem relevant to our analysis when we answer the questions: id, price, year, manufacturer, model, condition, cylinders, fuel, odometer, title status, transmission, drive, type, and state. These will be included and saved in the main data table.
On the other hand, the following features are considered irrelevant: url, region, region url, VIN, size, paint color, image url, description, county, lat, long, and posting date. They will be included and saved in the sub data table
#### c. Dropping null values
By using a Pandas code, rows with null values are removed from the data frame. Null values basically mean missing data, which could cause error in machine learning model. After dropping such rows, the data table contains 127,232 rows.
#### d. Formatting the cylinders and year column 
For the **cylinders** column, we removed the object ‘cylinders’ and changed its data type to float 64 so that the column can be processed as numerical value. Also, cylinders values categorized as **other** were replaced with **0**. For the **Year** column, the numerical values were converted into string. 
#### e. Odometer
Focusing on the odometer column, we noticed the odometer of some cars is too high (See Chart 1). We would not expect the odometer of 200,000 miles or more, or nobody will not purchase used cars that have travelled such a long distance. Therefore, the used cars with odometer of more than 200,000 miles were excluded from the data table.
#### f. Year
Entry year varies in the data table. As is shown in Chart 2, some cars are more than 50 years old, which those who are looking for a used car would not like to drive it. We decided to just select the entry year 2001 or after. 
#### g. Price
The used car price widely varies as well (Chart 3). As a used car, we consider $100,000 or less to be reasonable and affordable. Additionally, we removed used car of $0 to avoid the data table from being skewed.
#### h. Conclusion of exploration
As a result of exploration, we decided to remove the cars that do not appear reasonable as used car, outlier, from our scope of prediction. We noted that the data table contains 103,685 rows after exploring odometer, year, and price columns. We plotted some visuals to find price outliers for that we compared different features against price.
An **outlier** can cause serious problems in statistical analyses. Outliers are values within a dataset that vary greatly from the others—they’re either much larger, or significantly smaller. Outliers may indicate variabilities in a measurement, experimental errors, or a novelty. Therefore, it is important to remove outliers.


### 2. Data analysis
#### a. Further review of year
During a further review, we look at the relationship between price and year. By creating a dot chart in Chart 4, we find that the newer the entry year, the more expensive the used car. However, there are some outliers in each year, which were removed from the data table. 

### 3. Database creation
During the exploratory and analysis phase, the following tables are created in SQL.
- Main table
- Sub table
- State table

### 4. Data preprocessing
#### a. Feature engineering and selection
	As a feature engineering, we recategorized the state and arranged them into four regions named as **west, mid-west, northeast, and south**. This process is expected to reduce the number of unique values in the state column. The following columns were included in the final dataset for the prediction: price, year, manufacturer, condition, cylinders, fuel, odometer, title status, transmission, drive, type, area. Based on our analysis conducted above on the data, these features were considered factors that would affect the used car price.
#### b. Splitting data
First, the final table was divided into a target, which is price, and features. Then each group was split into a dataset to be used to train the machine learning model and the other dataset to be used to test the machine learning model. We write a Python code for splitting dataset so that 90% of the dataset is assigned as training and 10% is assigned as testing.

### 5. Machine learning model choice
We selected the multiple linear regression as our machine learning model. In our understanding, the model is simple and easy to understand. In addition, the model would need less time to process the data for prediction compared with other models with more complexity.
A limitation would be in the number of variables the multiple linear regression model can handle. Unlike the decision tree regression model, which we had also tried to use for prediction, the multiple linear regression model will not manage a number of variables. However, due to its simplicity and understandability, we decided to select the multiple linear regression model. 

### 6. Machine learning model training
The training dataset was fit to the model and trained for used car price prediction. Then, the modes was applied to the testing dataset. We understand that there would be little room to train the model. However, we could further preprocess the original data to improve accuracy of the model.

### 7. Accuracy of machine learning model
To evaluate accuracy of the multiple linear regression model, we are going to calculate r square, a statistical measure in a regression model that determines how well the data fit the regression model.



----------------------------------------------------------------------------------------------------------------------------------



## Assumptions set for Preprocessing

* Clean title, gas, and automatic transmission looks like a standard. 
* Odometer of the car may affect the upper limit of the used car price.


3. Preprocess
	i.	Entries in 2000 or older were removed.
	ii.	Odometer values larger than 200,000 (miles) were removed.
	iii.	Removed other data than the following
			a. Gas fuel
			b. Clean status
			c. Automatic transmission
	iV.	Outliers were removed as they were identified during the process.







* ### Worked on visualization to find Price Outliers
We plotted some visuals to find price outliers for that we compared different features against price. An **Outlier** can cause serious problems in statistical analyses. Outliers are values within a dataset that vary greatly from the others—they’re either much larger, or significantly smaller. Outliers may indicate variabilities in a measurement, experimental errors, or a novelty. Therefore its important to remove outliers.
--------------image will be added later-------------

* ### Removing Outliers
After visual interpretation, we realized that the lower 5% of the data has very low number of values and the upper 5% of the data was mainly very distinctive values. Therefore we decide to drop that portion of the data and set out Price range from 5th to 95th percentile. 

* ### One hot encoding
One hot encoding can be defined as the essential process of converting the categorical data variables to be provided to machine and deep learning algorithms which in turn improve predictions as well as classification accuracy of a model. We utized one hot encoding for converting our categorical features which are present in many of our columns like **fuel, manufacturer, model, condition, transmission, drive , etc**.

## Machine Learning Models Selection
We selected to work on:
* Decision Tree Regressor
* Linear Regression Model
* Lasso Model


### Libraries for data processing 

import numpy as np

import pandas as pd

### Libraries for visualization

import matplotlib.pyplot as plt

import plotly.express as px

import seaborn as sns

### Libraries for preprocessing

from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import PolynomialFeatures

from sklearn.preprocessing import StandardScaler,OneHotEncoder

### Liblaries for models

from sklearn.linear_model import LinearRegression, Ridge

from sklearn.tree import DecisionTreeRegressor

### Libraries for cross validation and model evaluation

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV, cross_val_score

### Libraries for SQL

import psycopg2

import sqlalchemy

from sqlalchemy.ext.automap import automap_base

from sqlalchemy.orm import Session

from sqlalchemy import create_engine, func

# Project-one
# Topic Selected:  Used Cars Price Prediction

## Background:


Over the past few years, the automotive industry has faced a shortage in the Semiconductor Integrate Chips globally. The Semicoductor IC is a critical component for controlling several electronic devices in the vehicle. 
Even though the car industry is growing at a fast rate, the shortage is impeding the growth of new car production and sales. The new car sales industry is making up for the hsortage by raising their APR and prices. Therefore, there is a trend of increasing demand of used cars which is making the prices of used cars higher as well.
Based on the current situation in the automotive industry, we have decided on predicting the prices of used cars for our project. Though it is a global issue, we will limit our studies and findings for US market only.


## Data Sources:
https://www.kaggle.com/code/maciejautuch/car-price-prediction/data
This dataset collected in Kaggle is mainly from craiglist.org (used item selling website) from all over US. These cars are from different manufacturers and of different years.

This Data has 26 columns and 426880 rows.

![image](https://user-images.githubusercontent.com/105535250/199819404-a0f16653-e9dd-437e-97a8-251c9d1c3d5b.png)

Our plan is to make Price as our target variable and rest ww will pass as features. Also we will be dropping off null values and some columns that are not needed as they dont impact the price of the used cars much.

https://public.tableau.com/authoring/Trial_16675235483750/Sheet1#1

## Required Libraries:

### Libraries for data processing 

import numpy as np

import pandas as pd


### Libraries for visualization

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


### Libraries for SQL

import psycopg2

import sqlalchemy

from sqlalchemy.ext.automap import automap_base

from sqlalchemy.orm import Session

from sqlalchemy import create_engine, func

## Questions to Answer:

1.	How does the mileage affect the price of the used car?
2.	How does size of the car impact the price?
3.	How does the age of the car, condition and fuel type affect the price of the car?
4.	Will this affect the overall demand for a used car in place of a new car for consumers?


## Quick review of Steps to be taken:

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

## Database
### Main Table
During the exploratory phase, we decided to select the following in the main tabel as they appears relevant to the used car price
	- a. id
	- b. price
	- c. year
	- d. manufacturer
	- e. model
	- f. condition
	- g. cylinders
	- h. fuel
	- i. odometer
	- j. title_status
	- k. transmission
	- l. drive
	- m. type	

### Sub Table
In a meantime, we decided to use the following as supplemental information.
	- a. id
	- n. url
	- o. region
	- p. region_url
	- q. VIN
	- r. paint_color
	- s. image_url
	- t. description
	- u. state
	- v. posting_date

Two tables are separately generated in PostgreSQL by creating connection to the database software. As an example, two tables are joined together as in picture 1 to create a new table including price, print_color, and image_url.

#### (Picture 1)

 <img src="https://github.com/lemleysamantha/Project-one/blob/main/Seg2_Database.png" width="320"/>  


## Data Exploration

We explored the data by creating box plots and subplots. We were able to better visualize the large amount of data that was given in the excel spreadsheet file. We found that the outliers did not make much sense. For examnple, Oregon had an average price of car at 1.6 million and California had an average price 139,000. Based on what we know about markets, California's average price should be way higher than Oregon's. We have found that our data also is not in one currency which could have skewed the numbers.

## Analysis

We were able to analyze the data by using the boxplots to visualize the outliers. The actual prices and the predicted prices were shown through the scatterplot.We were using tableau to get a better understanding of the prices and regions.


## Group Details:

We are team of four people. 


Shahla and Samantha are in charge of gathering information about the results of the dataset and what we want to achieve. Ryiochi was in charge of cleaning up the csv and dataset. Matthew was initializing our databases.

# SEGMENT 2 SQUARE ROLE

## Work Flow

* Data Collection
* Preprocessing/ Cleaning data
* Machine Learning Model Selection
* Split the data into Training and Testing 
* Fitting Data to a Model
* Evaluate the Model 

## Data Collection
We planned to work with pandas in jupyter notebook. For that we imported Panda Dependencies to create data frame. Data frames are more structured and tabular form and its more easier to process and analyze the data that way.

## Preprocessing/ Cleaning Data
We can not feed the raw data in the Machine learning model for that we worked on cleaning the data. 

* ### Dropped Unwanted Columns
![image](https://user-images.githubusercontent.com/105535250/199826222-7a26b31b-4c6f-410b-9473-4ca8bb55ca83.png)
In the image above you can see the name of columns we dropped out of total 26 columns. We made this decision because the information from these columns were not required to predict price for the Used Cars. 

* ### Dropped Null Values
Pandas DataFrame dropna() function is used to remove rows and columns with Null/NaN values which is basically missing data and it can cause error in Machine learning Model.

* ### Format the Cylinder and Year column 
For **cylinders** we changed the data type to float64 and remover the object cylinder to make it Numeric value also there were **257** cylinders categorized as **Others** we replace the value to **0**. For the **Year** column it was in decimal so we just changed the data type to integer to remove the decimal from this column. 

* ### Recategorize the State column
To reduce the number of unique values in the state column we recategorized the state and arranged them into four region named as **west, midwest, northeast, and south**.

* ### Worked on visualization to find Price Outliers
We plotted some visuals to find price outliers for that we compared different features against price. An **Outlier** can cause serious problems in statistical analyses. Outliers are values within a dataset that vary greatly from the others—they’re either much larger, or significantly smaller. Outliers may indicate variabilities in a measurement, experimental errors, or a novelty. Therefore its important to remove outliers.
--------------image will be added later-------------

* ### Removing Outliers
After visual interpretation, we realized that the lower 5% of the data has very low number of values and the upper 5% of the data was mainly very distinctive values. Therefore we decide to drop that portion of the data and set out Price range from 5th to 95th percentile. 

* ### One hot encoding
One hot encoding can be defined as the essential process of converting the categorical data variables to be provided to machine and deep learning algorithms which in turn improve predictions as well as classification accuracy of a model. We utized one hot encoding for converting our categorical features which are present in many of our columns like **fuel, manufacturer, model, condition, transmission, drive , etc**.

## Machine Learning Models Selection
We selected to work on:
* Decision Tree Regressor
* Linear Regression Model
* Lasso Model

## Split the Data and Target
We then fed the data in the Machine Learning Model and using the features of the dataset, we  split the processed data into training and testing data. We trained our machine learning algorithm with training data then we tested or evaluated our machine learning model with the test data.
first we created two variable **X** and **Y** to split data and the target. We stored **Price** in **Y** which is our target variable and pass rest of the features in varaible **X**. 

## Split Training and Testing data
for this we created four variables:

**X_train, X_test, Y_train, and y_test**
As we seperated the target from the data above, we then put all the data to train the module in the **X_train** variable and all the testing data in the variable **X_test**. The price of all the values from **X_train** will be stored in **y_train** and the price of all the values from **X_test** will be stored in **y_test**. 
We then utilized train-test-split function which we imported from sklearn library and pass our **X** and **Y** variable in it to finally split our data into traing and testing. 


