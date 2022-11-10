# Project-one
# Topic Selected:  Used Cars Price Prediction

## Background:

Over the past few years, the automotive industry has faced a shortage in the Semiconductor Integrate Chips globally. The Semicoductor IC is a critical component for controlling several electronic devices in the vehicle. 
Even though the car industry is growing at a fast rate, the shortage is impeding the growth of new car production and sales. The new car sales industry is making up for the hsortage by raising their APR and prices. Therefore, there is a trend of increasing demand of used cars which is making the prices of used cars higher as well.

## Objective:
Based on the current situation in the automotive industry, we have decided on predicting the prices of used cars for our project using the set of variables in the data set. Using Scikit learn to create following 2 different Machine Learning techniques:
* DecissionTress Regressor
* Linear Regression Model

Though it is a global issue, we will limit our studies and findings for US market only.

## Data Sources:
https://www.kaggle.com/code/maciejautuch/car-price-prediction/data
This dataset collected in Kaggle is mainly from craiglist.org (used item selling website) from all over US. These cars are from different manufacturers and of different years.

## Dataset:
* The original dataset contains 426,880 rows with 26 columns.
* Based on relevance of each column to our analysis and the number of available, i.e., not NaN, values in the column, decided to focus on the fullowing columns:
Price, year, manufacturer, model, condition, cylinders, fuel, odometer, title status, transmission, drive, and type
* Any rows that include NaN data have been removed from the dataset, which then contains 103577 rows with 12 columns.
		i)	Id, price, year, and odometer are numerical.
		ii)	Manufacturer, model, condition, cylinders, fuel, title status, transmission, drive, and type are object.
* Id and year could be converted into object and date, respectively.
* In addition, cylinders will be processed and used as number for further analysis.

![image](https://user-images.githubusercontent.com/105535250/199819404-a0f16653-e9dd-437e-97a8-251c9d1c3d5b.png)

Our plan is to make Price as our target variable and rest ww will pass as features. Also we will be dropping off null values and some columns that are not needed as they dont impact the price of the used cars much.

# Technologies, languages, tools, and algorithms used throughout project 
* ## Postgres pgAdmin (SQL)
PostgreSQL, also known as Postgres, is an open-source relational database with a strong reputation for its reliability, flexibility and support of open technical standards. PostgreSQL supports both non-relational and relational data types. pgAdmin is the community client for using PostgreSQL. It is usually installed along with PostgreSQL. While psql is a simple command-line tool, pgAdmin is a graphical user interface that provides pretty much the same functionality.

* ### Python
Python is a computer programming language often used to build websites and software, automate tasks, and conduct data analysis. Python is a general-purpose language, meaning it can be used to create a variety of different programs and isn't specialized for any specific problems.

* ### Pandas
Pandas is a fast, powerful, flexible and easy to use open source data analysis and manipulation tool,built on top of the Python programming language.

* ### Machine Learning Model
ML models are the mathematical engines of Artificial Intelligence, expressions of algorithms that find patterns and make predictions faster than a human can.

There are two types of Machine Learning:

* Supervised Machine Learning: It is an ML technique where models are trained on labeled data i.e output variable is provided in these types of problems. Here, the models find the mapping function to map input variables with the output variable or the labels.
- Regression and Classification problems are a part of Supervised Machine Learning.

* Unsupervised Machine Learning: It is the technique where models are not provided with the labeled data and they have to find the patterns and structure in the data to know about the data.
- Clustering and Association algorithms are a part of Unsupervised ML.
For our project we are using **Supervised Machine learning**.

* ### Tableau
Tableau Software is a tool that helps make Big Data small, and small data insightful and actionable. The main use of tableau software is to help people see and understand their data.

* ### Google Slide
Google Slides is an online presentation app that lets you create and format presentations and work with other people.

## Requirements for Machine Learning Model:
A Python library is a collection or package of various modules. It contains bundles of code that can be used repeatedly in different programs.

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

## Questions to Answer:

1.	How does the age of the car, condition and fuel type affect the price of the car?

2.	Will this affect the overall demand for a used car in place of a new car for consumers?

# Work Flow

* Data Collection
* Preprocessing/ Cleaning data
* Creating table in SQL
* Machine Learning Model Selection
* Split the data into Training and Testing 
* Fitting Data to a Model
* Evaluate the Model 
* Utilize Tableau and Google Slide for Visualization and Presentation

## Data Collection
We planned to work with pandas in jupyter notebook. For that we imported Panda Dependencies to create data frame. Data frames are more structured and tabular form and its more easier to process and analyze the data that way.

## Assumptions set for Preprocessing

* Assume that nobody would like to purchase cars that are more than 20 years old.
* Also assume that buyers would avoid cars that have already travelled more than 200000 miles
* Price trend: The newer the year of entry is, the higher the used car price is.
* Clean title, gas, and automatic transmission looks like a standard. 
* Odometer of the car may affect the upper limit of the used car price.


	iV.	Outliers were removed as they were identified during the process.

## Preprocessing/ Cleaning Data
We can not feed the raw data in the Machine learning model for that we worked on cleaning the data. 

* ### Dropped Unwanted Columns
![image](https://user-images.githubusercontent.com/105535250/199826222-7a26b31b-4c6f-410b-9473-4ca8bb55ca83.png)
In the image above you can see the name of columns we dropped out of total 26 columns. We made this decision because the information from these columns were not required to predict price for the Used Cars. 

* ### Dropped Null Values
Pandas DataFrame dropna() function is used to remove rows and columns with Null/NaN values which is basically missing data and it can cause error in Machine learning Model.

* ### Format the Cylinder and Year column 
For **cylinders** we changed the data type to float64 and remover the object cylinder to make it Numeric value also there were **257** cylinders categorized as **Others** we replace the value to **0**. For the **Year** column it was in decimal so we just changed the data type to integer to remove the decimal from this column. 

* ### Year Entries 
Considering the age of the cars can be an important variable, we tried to improve data by dropping the data in which car price is more than 20 years old setting Year Entries for 2000 or older were removed
![image](https://user-images.githubusercontent.com/105535250/201019725-cbfd5a53-9d7b-4aac-a23d-edbcbdfb0fdb.png)


* ### Odometer values 
Odometer values larger than 200,000 (miles) were removed.
![image](https://user-images.githubusercontent.com/105535250/201019426-821f0822-610b-4222-bf7e-28a9a29da39c.png)


* ### Recategorize the State column to Area
To reduce the number of unique values in the state column we recategorized the state and arranged them into four region named as **west, midwest, northeast, and south** given new column name as Area

![image](https://user-images.githubusercontent.com/105535250/201024725-5361b85f-e3e8-49b3-a1eb-2872a097cac8.png)


* ### Worked on visualization to find Price Outliers
An **Outlier** can cause serious problems in statistical analyses. Outliers are values within a dataset that vary greatly from the others—they’re either much larger, or significantly smaller. Outliers may indicate variabilities in a measurement, experimental errors, or a novelty. Therefore its important to remove outliers.

![image](https://user-images.githubusercontent.com/105535250/201022939-d19184a9-53ce-4a0a-a424-1aadb85b674c.png)

We plotted some visuals to find price outliers for that we compared Year features against price.

![image](https://user-images.githubusercontent.com/105535250/201024297-268a0e7b-e211-421e-baae-17ab657824ca.png)

* ### Removing Outliers
After visual interpretation, we realized that the lower 5% of the data has very low number of values and the upper 5% of the data was mainly very distinctive values. Therefore we decide to drop that portion of the data and set the range for price less then or equal to 100,000.

![image](https://user-images.githubusercontent.com/105535250/201024542-736df38f-625c-43c2-9a8b-b7b352525008.png)

## Database - Worked in pgAdmin
### Main Table
During the exploratory phase, we decided to select the following in the main tabel as they appears relevant to the used car price
	- a. id 		- h. fuel			
	- b. price		- i. odometer		
	- c. year		- j. title_status			
	- d. manufacturer	- k. transmission
	- e. model		- l. drive
	- f. condition		- m. type				
	- g. cylinders
	
	
	
	
	
		

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



* ### One hot encoding
One hot encoding can be defined as the essential process of converting the categorical data variables to be provided to machine and deep learning algorithms which in turn improve predictions as well as classification accuracy of a model. We utized one hot encoding for converting our categorical features which are present in many of our columns like **fuel, manufacturer, model, condition, transmission, drive , etc**.

## Machine Learning Models Selection
We selected to work on two Models i-e Decision Tree Regressor and Linear Regression Model. They are discussed below.

* ## Decision Tree Regressor
Decision Tree is one of the most commonly used, practical approaches for supervised learning. It can be used to solve both Regression and Classification tasks with the latter being put more into practical application. It is used by the Train Using AutoML tool and classifies or regresses the data using true or false answers to certain questions. The resulting structure, when visualized, is in the form of a tree with different types of nodes—root, internal, and leaf.

![image](https://user-images.githubusercontent.com/105535250/201030320-ce757d60-9499-4c49-82fc-737b211c345f.png)

### Advantages of Decission Tree Regressor

There are many advantages of this model. some of them are:

1- Compared to other algorithms decision trees requires less effort for data preparation during pre-processing.

2- A decision tree does not require normalization of data.

3- A decision tree does not require scaling of data as well.

4- Missing values in the data also do NOT affect the process of building a decision tree to any considerable extent.

5- A Decision tree model is very intuitive and easy to explain to technical teams as well as stakeholders and can be used for both classification and regression problems.

### Disadvantages of Decission Tree Regressor

1- A small change in the data can cause a large change in the structure of the decision tree causing instability.

2- Decision tree often involves higher time to train the model.

3- Decision tree training is relatively expensive as the complexity and time has taken are more.

4- The Decision Tree algorithm is inadequate for applying regression and predicting continuous values.

5- It can’t be used in big data: If the size of data is too big, then one single tree may grow a lot of nodes which might result in complexity and leads to overfitting.

* ## Linear Regression Model
Linear Regression may be one of the most commonly used models in the real world. It is a linear approach to modeling the relationship between a scalar response (dependent variable) and one or more explanatory variables (independent variables). Linear regression is used in everything from biological, behavioral, environmental and social sciences to business.

![image](https://user-images.githubusercontent.com/105535250/201041695-9db95289-3668-4514-8bee-69d5ad936108.png)

### Advantages of Linear Regression

1. Linear Regression performs well when the dataset is linearly separable. We can use it to find the nature of the relationship among the variables.

2. Linear Regression is easier to implement, interpret and very efficient to train. 

3. Linear Regression is prone to over-fitting but it can be easily avoided using some dimensionality reduction techniques, regularization (L1 and L2) techniques and cross-validation.

### Disadvantages of Linear Regression

1. Main limitation of Linear Regression is the assumption of linearity between the dependent variable and the independent variables. In the real world, the data is rarely linearly separable. It assumes that there is a straight-line relationship between the dependent and independent variables which is incorrect many times.

2. Prone to noise and overfitting: If the number of observations are lesser than the number of features, Linear Regression should not be used, otherwise it may lead to overfit because is starts considering noise in this scenario while building the model.

3. Prone to outliers: Linear regression is very sensitive to outliers (anomalies). So, outliers should be analyzed and removed before applying Linear Regression to the dataset.

4. Prone to multicollinearity: Before applying Linear regression, multicollinearity should be removed (using dimensionality reduction techniques) because it assumes that there is no relationship among independent variables.

## Split the Data and Target
We then fed the data in the Machine Learning Model and using the features of the dataset, we  split the processed data into training and testing data. We trained our machine learning algorithm with training data then we tested or evaluated our machine learning model with the test data.
first we created two variable **X** and **Y** to split data and the target. We stored **Price** in **Y** which is our target variable and pass rest of the features in varaible **X**. 

## Split Training and Testing data
for this we created four variables:

**X_train, X_test, Y_train, and y_test**
As we seperated the target from the data above, we then put all the data to train the module in the **X_train** variable and all the testing data in the variable **X_test**. The price of all the values from **X_train** will be stored in **y_train** and the price of all the values from **X_test** will be stored in **y_test**. 
We then utilized train-test-split function which we imported from sklearn library and pass our **X** and **Y** variable in it to finally split our data into traing and testing. 
