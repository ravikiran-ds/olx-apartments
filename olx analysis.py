# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 21:25:57 2020

@author: HP
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("data.csv")
#lokkin at data
df.head()
df.tail()
#shape of the data
row,col=df.shape
#no missing data
df.isnull().sum()
#datatypes
df.dtypes

#exploring
#separating state
df['state']=df.location.apply(lambda x:x.split(',')[-1])
#separating city
df['city']=df.location.apply(lambda x:x.split(',')[-2])


#bar plots
def bar_plot(df,col,title,xlab,ylab,xlim=None):
    df[col].value_counts().plot.bar()
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.xlim(xlim)

#construction status
#6 different stages of construction
len(df['construction_status'].unique())
bar_plot(df,"construction_status","Construction Status","Status","Buildings")
#284 apartments Ready to move in and 182 under construction
df['construction_status'].value_counts()

#furnishing
#5 different stages of furnishing
len(df['furnishing'].unique())
bar_plot(df,'furnishing','Furnishing','Status','Apartments')
#254 unfurnished , 166 semi furnished , 68 furnished
df['furnishing'].value_counts()

#city
#apartments from 68 different cities
len(df['city'].unique())
#top 10 cities 
bar_plot(df,'city','City',"City",'Apartments',xlim=(0,10))
#100 in mumbai,65 in pune
df['city'].value_counts()

#state
#20 diff states
len(df['state'].unique())
#top 10 states
bar_plot(df,'state','State',"Name",'Apartments',xlim=(0,10))
#229 9in maharastra,52 in kerala
df['state'].value_counts()

#hist plots
def hist_plot(df,col,title,xlab,ylab,color=None,xlim=None):
    df[col].hist(color=color)
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.xlim(xlim)

#bathrooms    
#most apartments have 2 bathrooms
hist_plot(df,'bathrooms','Bathrooms','# of bathrooms','Apartments',color='blue')

#built-up area
#most apartments are around 500 - 1500 ft built up area
hist_plot(df,'built_up_area',"Built up area",'Area','Apartments',color='green')

#carpet area
#many are between 0-2000 ft
hist_plot(df,'carpet_area','Carpet Area','Area','Apartments',color='black',xlim=(0,5000))
#one house with 20000 ft area
df['carpet_area'].plot.box()
plt.title("Carpet Area")
plt.xlabel("Area")
plt.ylabel("in ft2")

#price
#many apartments lie in range 0-1,00,00,000
hist_plot(df,'price',"Price",'Rupees','Apratments',color='brown')
#outliers
df['price'].plot.box()
plt.title("Price")
plt.ylabel("in Rupees")

#bedrooms
#max apartments have 2 bedrooms
hist_plot(df,'bedrooms',"Bedrooms",'Rooms','Apartments',color='pink')


#feature engineering
df['bedrooms']=np.where(df['bedrooms']=='4+',5,df['bedrooms'])
df['bathrooms']=np.where(df['bathrooms']=='4+',5,df['bathrooms'])
df['bedrooms']=df['bedrooms'].astype('int')
df['bathrooms']=df['bathrooms'].astype(int)

#df.dtypes
df2=df.copy()
df2=df2.drop(['description','location','state'],axis=1)
#correlation
sns.heatmap(df2.corr(),annot=True)
#highly correlated data

#predicting price
x=df.drop(['price','description','location','state'],axis=1)
y=df['price'].values
y=y.reshape(-1,1)

#getting dummies fro categorical variables
x=pd.get_dummies(x)

#splitting training test dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=101)

#normalizing the data
from sklearn.preprocessing import MinMaxScaler
scl=MinMaxScaler()
x_train=scl.fit_transform(x_train)
x_test=scl.transform(x_test)

#using random forest regressor
from sklearn.ensemble import RandomForestRegressor
rfreg=RandomForestRegressor(random_state=101)
#fitting
rfreg.fit(x_train,y_train)

#predicting
y_pred=rfreg.predict(x_test)





