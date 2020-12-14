# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 17:40:04 2020

@author: Deepak
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
cal = pd.read_csv("H:\DATA SCIENCE\Modules\Module 6 Simpal linear regression\DataSets/calories_consumed.csv")
cal
# indepandent variable = 
plt.scatter (x=cal['Weightgainedgrams'], y=cal['CaloriesConsumed'], color = 'Red')
np.corrcoef(cal.Weightgainedgrams, cal.CaloriesConsumed)

# 1st moment business Decision
cal.mean()
cal.mode()
cal.median()

# 2nd moment business Decision
np.std(cal['Weightgainedgrams'])
np.std(cal['CaloriesConsumed'])
np.var(cal['Weightgainedgrams'])
np.var(cal['CaloriesConsumed'])

range =  max(cal['Weightgainedgrams'])-min(cal['Weightgainedgrams'])
range
range =  max(cal['CaloriesConsumed'])-min(cal['CaloriesConsumed'])
range 
# 3rd moment business Decision Skewness
cal.skew()
cal['Weightgainedgrams'].skew() # Skew is +ve  mean > median  
cal['CaloriesConsumed'].skew()

# 4th moment business Decision Kurtosis
cal.kurt()
cal["Weightgainedgrams"].kurt() # 
cal['CaloriesConsumed'].kurt()

# using sklearn library
# %matplotlib inline
# Visulize data
plt.xlabel("X-axis CaloriesConsumed_Independent")
plt.ylabel("Y-axis Weightgainedgrams_Dependent")
plt.scatter(cal.CaloriesConsumed, cal.Weightgainedgrams, color = 'red', marker = '+')
np.corrcoef(cal.Weightgainedgrams, cal.CaloriesConsumed)

# model 1
import statsmodels.formula.api as smf 

model = smf.ols('Weightgainedgrams ~ CaloriesConsumed' , data=cal).fit() # (y/output ~ x/input)
model.summary() # R^2 =0.897

pred1 = model.predict(pd.DataFrame(cal['CaloriesConsumed'])) # predict 
pred1
print (model.conf_int(0.05)) # 95% confidence interval

res = cal.Weightgainedgrams  - pred1 #weight gain - predict1 = res
res
sqres = res*res # takong square
sqres
mse = np.mean(sqres) # taking mean of res^2
mse
rmse = np.sqrt(mse) # RMSE= square of mse1q
rmse   #103.3025

######### Model building on Transformed Data

# Log Transformation
# x = log(Weightgainedgrams); y = at
plt.scatter(x=np.log(cal['CaloriesConsumed']),y=cal['Weightgainedgrams'],color='brown')
np.corrcoef(np.log(cal.CaloriesConsumed), cal.Weightgainedgrams) #correlation =0.8987

model2 = smf.ols('Weightgainedgrams ~ np.log(CaloriesConsumed)',data=cal).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(cal['CaloriesConsumed']))
pred2
print(model2.conf_int(0.01)) # 99% confidence level

res2 = cal.Weightgainedgrams - pred2
res2
sqres2 = res2*res2
mse2 = np.mean(sqres2)
rmse2 = np.sqrt(mse2)
rmse2
# Exponential transformation
plt.scatter(x=cal['CaloriesConsumed'], y=np.log(cal['Weightgainedgrams']),color='orange')

np.corrcoef(cal.CaloriesConsumed, np.log(cal.Weightgainedgrams)) #correlation

model3 = smf.ols('np.log(Weightgainedgrams) ~ CaloriesConsumed',data=cal).fit()
model3.summary()

pred_log = model3.predict(pd.DataFrame(cal['CaloriesConsumed']))
pred_log
pred3 = np.exp(pred_log)
pred3
print(model3.conf_int(0.01)) # 99% confidence level

res3 = cal.Weightgainedgrams - pred3
sqres3 = res3*res3
mse3 = np.mean(sqres3)
rmse3 = np.sqrt(mse3)
rmse3

from sklearn.model_selection import train_test_split
X = cal[["CaloriesConsumed"]]
X
y = cal["Weightgainedgrams"]
y
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3)
X_train
y_test

import sklearn
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
lab = LinearRegression()
lab.fit(X_train,y_train)

lab.predict(X_test)
X_test
y_test
lab.score(X_train,y_train)
lab.score(X_test,y_test)
