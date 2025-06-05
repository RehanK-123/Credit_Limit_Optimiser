import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
from random import uniform
from sklearn.preprocessing import LabelEncoder

def random_salary(range):
    """This function takes in a string, parses it, and then uses the upper limit and lower limit of the range to randomly
    return a value within that range."""

    a, b = [], ""
    for i in range:
        if i.isdigit():
            b += i
        
        else:
           if b: a.append(b)
           b = ""

    print(a, b)    
    return uniform(int(a[0]) * 1000, int(a[1]) * 1000)    

#loading the dataset     
data = pd.read_csv("Credit_Card_Dataset_New.csv")
for i in data.columns:
    print(data[i].unique())
#encoding the binary values of the Pay_on_time col with 1 and 0
encoder = LabelEncoder()
data["Pay_on_time"] = encoder.fit_transform(data["Pay_on_time"])

list = []
#randomly assigning a salary value from the range given 
data["Income_Category"] = data["Income_Category"].replace({"<30k":"0-30k", "111k+":"111-150k","!NULL":np.nan}) 
for row in range(data["Income_Category"].shape[0]):
    if data["Income_Category"][row] == "#NULL!":
        list.append(row)
        data.drop(row, inplace= True)
        
    else:
        data["Income_Category"][row] = random_salary(data["Income_Category"][row])
        
#re-writing the dataset into our file
# print(list)
# data.to_csv("Credit_Card_Dataset.csv")

