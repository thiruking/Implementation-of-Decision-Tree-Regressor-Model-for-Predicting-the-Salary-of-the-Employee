# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the libraries and read the data frame using pandas.

2.Calculate the null values present in the dataset and apply label encoder.

3.Determine test and training data set and apply decison tree regression in dataset.

4.Calculate Mean square error,data prediction and r2.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: THIRUMALAI K
RegisterNumber:  212224240176
*/
```
```
import pandas as pd
data=pd.read_csv("/content/Salary.csv")

data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
y=data["Salary"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)


from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
```

## Output:
### Reading of dataset


![image](https://github.com/user-attachments/assets/327b23d0-86c8-4de7-8c0a-84b5b108d834)

### Value of df.head()

![image](https://github.com/user-attachments/assets/07593d6f-d751-4ff8-9702-57b5499be1c4)

### Value of df.isnull().sum()
![image](https://github.com/user-attachments/assets/db92ec36-3f7e-4224-bf76-214efafe936d)

### df.info()

![image](https://github.com/user-attachments/assets/ad80bc6c-27e9-4431-ad02-2bd3a264be79)

Data after encoding calculating Mean Squared Error

![image](https://github.com/user-attachments/assets/b28f17f2-5a72-4e54-8d19-6bb3bbbfff76)

R2 value
![image](https://github.com/user-attachments/assets/781ebba9-120b-4d65-b2a0-f6e8a99a5d1f)

Model prediction with [5,6] as input
![image](https://github.com/user-attachments/assets/e8194090-6b27-4fdb-b4ed-539241954e1b)












## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
